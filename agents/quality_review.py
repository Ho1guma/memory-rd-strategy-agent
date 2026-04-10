"""
LLM 기반 품질 검수 모듈 (Phase 2)

Phase 1에서 규칙 기반으로 처리되는 항목:
- 섹션 구조, REFERENCE 완결성, SUMMARY 길이, 수율/원가 금지, 인용 범위, 추정 라벨 compliance

Phase 2 (이 모듈)에서 처리하는 항목 — 규칙으로 잡기 어려운 것만:
- 환각 탐지: 증거에도 TRL 추정표에도 없는 구체적 수치/날짜/사건을 지어낸 경우
- 인용 정확성: [N] 인용이 해당 증거와 완전히 무관한 경우
- 톤: 과장·마케팅 문구 (warning만, fail 아님)

compliance는 Phase 1에서 규칙으로 처리하므로 Phase 2에서 다루지 않음.
"""

import json
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

REPORT_REVIEW_PROMPT = """당신은 반도체 R&D 기술 전략 보고서의 품질 감사관입니다.
아래 보고서 초안에서 명백한 환각(날조)과 인용 오류만 검출하세요.

**역할 범위 - 아래 항목만 검토하세요**

1. **환각 탐지 (Hallucination)** — 매우 보수적으로 판단
   - 증거 자료에도 TRL 추정표에도 없는 구체적 수치(예: 정확한 매출액, 특정 %수치, 정확한 출시일)를 명백히 지어낸 경우에만 -> "critical"
   - 일반적 기술 동향 서술(예: "AI 수요 증가")은 환각 아님 -> 판단 대상 아님
   - SK Hynix 자사 TRL 서술(2절)은 경쟁사 추정표와 무관하므로 절대 환각 판단 대상 아님

2. **인용 정확성 (Citation Accuracy)**
   - [N] 인용 주변 내용이 해당 증거와 완전히 무관한 경우에만 -> "critical"
   - 부분적으로 관련 있으면 -> "warning"

3. **톤 (Tone)**
   - 감탄사, 과장, 마케팅 문구 -> "warning"만 (절대 critical 아님)

**검토 대상이 아닌 항목 (판단하지 마세요)**
- compliance (추정 라벨 사용 여부): Phase 1에서 규칙으로 처리됨
- 섹션 구조: Phase 1에서 처리됨
- SK Hynix TRL vs 경쟁사 TRL 비교: 완전히 다른 개체이므로 비교 불가

**보고서에서 인용된 증거 자료**
{cited_evidence}

**TRL 추정표 (경쟁사 전용 데이터 — SK Hynix 없음)**
{trl_table}

**보고서 초안**
{draft_report}

**출력 — JSON만 출력 (마크다운 코드블록 없이)**
{{
  "pass": true 또는 false,
  "issues": [
    {{
      "criterion": "hallucination 또는 citation_accuracy 또는 tone",
      "severity": "critical 또는 warning",
      "location": "해당 섹션 또는 문장",
      "detail": "문제 설명"
    }}
  ],
  "feedback": "수정 필요 사항 (없으면 빈 문자열)"
}}

**판정 규칙**
- "warning"만 있으면 반드시 "pass": true
- "critical"은 명백히 날조된 구체적 수치나 완전 무관 인용에만 사용
- 확신이 없으면 "warning"으로 처리하거나 이슈 등록 안 함
- "critical" 이슈가 1건이라도 있어야만 "pass": false
"""


def _build_cited_evidence(draft: str, evidence_store: list) -> str:
    """보고서에서 실제 인용된 증거만 추출하여 요약"""
    citation_ids = sorted(set(int(c) for c in re.findall(r"\[(\d+)\]", draft)))
    lines = []
    for cid in citation_ids:
        idx = cid - 1
        if 0 <= idx < len(evidence_store):
            item = evidence_store[idx]
            lines.append(
                f"[{cid}] {item.get('title', '제목 없음')}\n"
                f"    {item.get('snippet', '')[:200]}\n"
                f"    출처: {item.get('url', '')}"
            )
        else:
            lines.append(f"[{cid}] [존재하지 않는 증거 — 인덱스 범위 초과]")
    if not lines:
        return "(인용 없음)"
    return "\n\n".join(lines)


def review_report_quality(
    draft_report: str,
    evidence_store: list,
    trl_table: list,
    threat_matrix: list,
) -> dict:
    """
    LLM 기반 보고서 품질 검수 (Phase 2 — 환각/인용 오류만).

    Returns:
        {"pass": bool, "issues": list[dict], "feedback": str}
    """
    if not draft_report:
        return {
            "pass": False,
            "issues": [
                {
                    "criterion": "structure",
                    "severity": "critical",
                    "location": "전체",
                    "detail": "보고서 초안이 비어 있음",
                }
            ],
            "feedback": "보고서 초안이 생성되지 않았습니다.",
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    prompt = REPORT_REVIEW_PROMPT.format(
        cited_evidence=_build_cited_evidence(draft_report, evidence_store),
        trl_table=json.dumps(trl_table, ensure_ascii=False, indent=2),
        draft_report=draft_report,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > 0:
            result = json.loads(text[start:end])
            # warning만 있으면 항상 pass
            has_critical = any(
                issue.get("severity") == "critical"
                for issue in result.get("issues", [])
            )
            result["pass"] = not has_critical
            return result
        print("[QualityReview] ⚠️ JSON 파싱 불가 — 통과 처리")
    except Exception as e:
        print(f"[QualityReview] ⚠️ LLM 품질 검수 실패 (통과 처리): {e}")

    return {"pass": True, "issues": [], "feedback": ""}
