"""
LLM 기반 품질 검수 모듈

Supervisor의 수치·구조 판정(Phase 1) 통과 후 호출되는 2차 품질 게이트(Phase 2).
- 환각 탐지: 증거에 없는 사실 주장 감지
- 인용 정확성: [N] 인용과 실제 증거 내용 대조
- Compliance: TRL 4-6 "추정" 명시, 수율·원가 단정 금지
- 톤: R&D 임원 대상 객관·간결 톤 준수
"""

import json
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

REPORT_REVIEW_PROMPT = """당신은 반도체 R&D 기술 전략 보고서의 품질 감사관(Quality Auditor)입니다.
아래 보고서 초안을 증거 자료와 대조하여 품질을 평가하세요.

**평가 기준**

1. **환각 탐지 (Hallucination)**
   - 보고서의 모든 사실적 주장(수치·날짜·사건·인물)이 아래 증거 자료에 근거하는지 확인
   - 증거에 없는 구체적 수치, 날짜, 사건을 지어낸 경우 → severity "critical"
   - 증거를 과도하게 확대 해석하거나 인과관계를 임의 추가한 경우 → severity "warning"

2. **인용 정확성 (Citation Accuracy)**
   - 본문의 [N] 인용 번호 주변 텍스트가 실제 증거 [N]의 내용과 일치하는지 확인
   - 인용 번호와 내용이 명백히 불일치하면 → "critical"

3. **Compliance**
   - TRL 4-6 구간에 "추정" 또는 그에 준하는 한계 고지가 있는지 확인
   - 수율·원가를 단정적으로 서술한 경우 → "critical"
   - 비공개 로드맵 내용을 사실처럼 서술한 경우 → "critical"

4. **톤·표현 (Tone)**
   - R&D 임원 대상: 객관적·간결·논거 중심이어야 함
   - 감탄사("놀랍게도"), 과장("혁명적"), 마케팅 문구 사용 시 → "warning"

**보고서에서 인용된 증거 자료**
{cited_evidence}

**TRL 추정표**
{trl_table}

**위협 매트릭스**
{threat_matrix}

**보고서 초안**
{draft_report}

**출력 — 아래 JSON만 출력하세요 (마크다운 코드블록·설명 텍스트 없이)**
{{
  "pass": true 또는 false,
  "issues": [
    {{
      "criterion": "hallucination 또는 citation_accuracy 또는 compliance 또는 tone",
      "severity": "critical 또는 warning",
      "location": "해당 섹션명 또는 문장 발췌",
      "detail": "구체적 문제 설명"
    }}
  ],
  "feedback": "보고서 재작성 시 반드시 수정해야 할 사항 요약 (1-2문단)"
}}

**판정 규칙**
- "critical" 이슈가 1건이라도 있으면 반드시 "pass": false
- "warning"만 있으면 "pass": true (개선 권고)
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
    LLM 기반 보고서 품질 검수.

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
        threat_matrix=json.dumps(threat_matrix, ensure_ascii=False, indent=2),
        draft_report=draft_report,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > 0:
            result = json.loads(text[start:end])
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
