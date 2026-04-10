"""
Report Agent (T6, T7)
- T6: TRL 표 + 위협 매트릭스 → 보고서 목차 4절 초안 생성
- T7: REFERENCE 정합성 검증 + SUMMARY 압축
- 출력: draft_report (Markdown), reference_list
"""

import json
import os
import re
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agents.state import AgentState, ReferenceItem


LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


REPORT_PROMPT = """당신은 SK Hynix R&D 부서의 기술 전략 보고서 작성 전문가입니다.

**독자 (Target Actor)**
- SK Hynix R&D 담당자 및 기술 전략 임원 (반도체 회로·공정·시스템 고급 전문가)
- 톤: 객관적·간결·논거 중심 (감탄사·수사적 표현 지양)
- 문장 길이: 1문장 최대 2줄 이내, 핵심만 기술
- SUMMARY는 A4 반 페이지(약 400자) 이내로 압축

**분석 범위**
- 기술: {technologies}
- 경쟁사: {competitors}

**TRL 추정표**
{trl_table}

**위협 수준 매트릭스**
{threat_matrix}

**수집 증거 요약 ({evidence_count}건)**
{evidence_summary}

**보고서 목차 (반드시 아래 섹션명 그대로 사용)**

SUMMARY
(보고서 전체 핵심 내용 요약, 400자 이내)

1. 앞페이지
SK Hynix 내부 보고서 포맷에 맞는 표지 정보 (보고서명, 작성일, 대상 독자, 보안 등급)

2. 분석 배경
왜 지금 이 기술을 분석해야 하는가
- 2.1 시장 동향 및 배경: 메모리 수요 추이, AI 가속기 시장 규모, 공급망·표준·정책 동향 (텍스트 문단 형식)
- [도표] 메모리 수요량 추이 또는 AI 서버 시장 성장 관련 수치 제시 (증거 기반, 수치 임의 생성 금지)

3. 분석 대상 기술 현황 — SK Hynix
SK Hynix의 HBM / PIM / CXL 각 도메인별 현재 기술 수준 및 개발 방향
- 3.1 HBM: 영업이익·시장 점유율·연구 개발 수준 (TRL 포함)
- 3.2 PIM: 영업이익·시장 점유율·연구 개발 수준 (TRL 포함)
- 3.3 CXL: 영업이익·시장 점유율·연구 개발 수준 (TRL 포함)

4. 경쟁사 동향 분석
경쟁사의 기술 전략과 최신 움직임 — 기술 도메인별 구성
- 4.1 HBM
  - 경쟁사 비교표 (SK Hynix 포함, TRL·위협 등급 명시)
  - 주요 경쟁사별 세부 내용 (증거가 충분한 경우만 작성, 없으면 생략)
  - 비주류 업체 동향 (선택)
- 4.2 PIM (동일 구조)
- 4.3 CXL (동일 구조)

5. 장기 관점 신기술 동향 (선택)
차세대 메모리 개발 동향, 신개념 메모리-연산 통합 아키텍처 등
(증거 자료에 근거가 있는 경우에만 작성; 근거 없으면 섹션 생략)

6. 전략적 시사점
SK Hynix R&D 우선순위 관점 대응 방향 제언
- 단기 (1-2년): 즉시 실행 가능한 R&D 과제
- 중기 (3-5년): 기술 포지셔닝 및 투자 방향

REFERENCE
[1] 제목, URL, 접근일
(본문 인용 번호와 1:1 대응)

**Compliance 규칙**
- TRL 4-6 구간은 "추정" 명시 + 간접 지표(특허 출원 패턴·학회 빈도·채용 키워드) 근거 서술 필수
- 수율·원가·미공개 로드맵은 단정 금지, "비공개로 추정" 표현 사용
- 증거 자료에 없는 내용을 창작하거나 수치를 임의로 생성하지 말 것
- 섹션 5는 증거 기반 근거가 없으면 작성하지 말 것

**Formatting 규칙**
- Markdown 형식 출력
- 회사명 표기 통일: SK Hynix, Samsung, Micron, Intel, AMD
- 기술 용어는 한국어(영문 병기): 예) 대역폭(Bandwidth)
- 본문 내 출처 인용: [숫자] 형식
- 경쟁사 비교표는 Markdown 표 형식으로 작성
"""


def _extract_references(draft: str, evidence_store: list) -> list[ReferenceItem]:
    citation_ids = re.findall(r'\[(\d+)\]', draft)
    unique_ids = sorted(set(int(c) for c in citation_ids))
    today = date.today().isoformat()

    refs: list[ReferenceItem] = []
    for cid in unique_ids:
        idx = cid - 1
        if 0 <= idx < len(evidence_store):
            item = evidence_store[idx]
            refs.append(ReferenceItem(
                citation_id=f"[{cid}]",
                url=item["url"],
                title=item["title"],
                accessed_date=today,
            ))
        else:
            # Compliance: 매핑 불가 인용은 명시적으로 표기
            refs.append(ReferenceItem(
                citation_id=f"[{cid}]",
                url="",
                title=f"[매핑 불가 — 원본 출처 확인 필요]",
                accessed_date=today,
            ))
    return refs


def _build_evidence_summary(evidence_store: list, max_items: int = 20) -> str:
    lines = []
    for i, item in enumerate(evidence_store[:max_items]):
        lines.append(f"[{i+1}] {item['title']} — {item['snippet'][:150]}")
    return "\n".join(lines)


def report_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=8192)

    scope = state.get("scope", {})
    trl_table = state.get("trl_table", [])
    threat_matrix = state.get("threat_matrix", [])
    evidence_store = state.get("evidence_store", [])

    existing_draft = state.get("draft_report", "")
    quality_feedback = state.get("last_error", "")
    is_retry = bool(existing_draft and quality_feedback)

    print(f"[Report] 보고서 {'재작성' if is_retry else '초안 생성'} 중...")
    prompt = REPORT_PROMPT.format(
        technologies=", ".join(scope.get("technologies", [])),
        competitors=", ".join(scope.get("competitors", [])),
        trl_table=json.dumps(trl_table, ensure_ascii=False, indent=2),
        threat_matrix=json.dumps(threat_matrix, ensure_ascii=False, indent=2),
        evidence_count=len(evidence_store),
        evidence_summary=_build_evidence_summary(evidence_store),
    )

    if is_retry:
        prompt += (
            "\n\n**[품질 검수 피드백 — 아래 지적사항을 반드시 반영하여 보고서를 수정하세요]**\n"
            f"{quality_feedback}"
        )

    # Error Handling: API 오류 포착
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        draft = response.content
    except Exception as e:
        print(f"[Report] ⚠️ LLM 호출 실패: {e}")
        return {"draft_report": "", "reference_list": [], "last_error": str(e)}

    reference_list = _extract_references(draft, evidence_store)

    # Compliance: 매핑 불가 인용 경고
    unmapped = [r for r in reference_list if "매핑 불가" in r["title"]]
    if unmapped:
        print(f"[Report] ⚠️ 매핑 불가 인용 {len(unmapped)}건 — 환각 인용 가능성 검토 필요: {[r['citation_id'] for r in unmapped]}")

    print(f"[Report] 보고서 생성 완료 (REFERENCE {len(reference_list)}건)")

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/report_{date.today().isoformat()}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(draft)
    print(f"[Report] 저장 완료: {output_path}")

    return {
        "draft_report": draft,
        "reference_list": reference_list,
    }
