"""Report Agent — Tasks T6, T7.

T6 Outcome: Draft report with required 4 sections + SUMMARY + REFERENCE.
T7 Outcome: REFERENCE ↔ in-text citation consistency verified, SUMMARY compressed.

Each report section retrieves targeted evidence via hybrid_search rather than
using the full evidence_store.
"""
from __future__ import annotations

import json
import re
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rd_strategy_agent.state import AgentState, ReferenceItem
from rd_strategy_agent.agents.retrieve import hybrid_search

REPORT_SYSTEM = """You are a senior R&D strategy analyst at SK Hynix writing a comprehensive internal intelligence report.

Write a detailed Korean-language R&D strategy report in Markdown. Each section must be thorough — this is an executive-level document.

---

## SUMMARY
- 핵심 발견사항 5개 이상 bullet
- SK Hynix 최우선 행동 권고안 2~3줄

## 1. 분석 배경
시장 환경과 분석 목적을 2~3 단락으로 상세 서술:
- AI/HPC 수요 급증이 메모리 아키텍처에 미치는 구조적 변화
- HBM4·PIM·CXL이 동시에 부상하는 이유와 상호 관계
- SK Hynix 관점에서의 분석 목적 및 범위

## 2. 분석 대상 기술 현황

각 소절은 반드시 아래 구조를 따를 것. 분량: 소절당 최소 600자.

### 2-1. HBM4 (High Bandwidth Memory 4)

**기술 개요 및 성숙도**
- 핵심 스펙, 전작 대비 개선점, 현재 TRL 및 근거

**경쟁사 개발 현황** (각 기업을 아래 형식으로 번호 작성)

1. **Samsung**
   - 개발 중인 제품·공정 세대, 구체적 투자 규모 또는 공시 내용 [N]
   - 채용 동향 (JD 키워드, 채용 규모 신호) [N]

2. **Micron**
   - 개발 중인 제품·공정 세대, 구체적 투자 규모 또는 공시 내용 [N]
   - 채용 동향 [N]

3. **Intel**
   - HBM 수요·채택 측면 동향 [N]

4. **AMD**
   - HBM 수요·채택 측면 동향 [N]

**SK Hynix 포지션**
- 당사 제품 현황, 시장 점유율, 강점·약점 분석 [N]

**기술적 병목 및 도전 과제**
- 열 방출, TSV 수율, 원가 등 구체적 이슈

---

### 2-2. PIM (Processing-In-Memory)
(동일 구조 — 경쟁사 번호 목록 포함)

### 2-3. CXL (Compute Express Link)
(동일 구조 — 경쟁사 번호 목록 포함)

---

## 3. 경쟁사 위협 분석

**TRL 비교표** (Markdown table, company × technology):

| 기업 | HBM4 TRL | PIM TRL | CXL TRL | 종합 위협도 |
|------|----------|---------|---------|------------|
| ...  | ...      | ...     | ...     | ...        |

**위협 매트릭스** (Markdown table):

| 기업 | 위협 수준 | 핵심 근거 | 투자·채용 신호 |
|------|----------|----------|--------------|
| ...  | ...      | ...      | ...          |

각 경쟁사별 위협 요인을 2~3문장으로 추가 서술.

---

## 4. 전략적 시사점

**단기 (0~1년)**
- SK Hynix가 즉시 착수해야 할 R&D 액션 3개 이상, 각각 근거 포함 [N]

**중기 (2~3년)**
- 포트폴리오 투자 방향, 파트너십·M&A 고려 사항

**장기 / 신규 개념 (3~5년+)**
- 현재 연구 단계이지만 선점이 필요한 차세대 기술 컨셉 3개 이상
  (예: CXL 3.0/4.0 Fabric, Neuromorphic PIM, 3D-stacked Compute Memory, In-Memory AI Accelerator 등)
- 각 컨셉별 경쟁사 선점 리스크 및 SK Hynix 대응 방향

---

## REFERENCE
[N] Title. URL. Accessed YYYY-MM-DD.

---

IMPORTANT writing rules:
- 모든 사실적 주장에 [N] 인용 필수 — 인용 없는 단락 금지. 가능한 한 많은 source를 인용할 것.
- 제공된 sources ID만 사용 (없는 번호 금지)
- TRL 4~6은 "(추정)" 표기 + 간접 지표 명시
- 기업별 개발 현황은 반드시 번호 목록(1. 2. 3. 4.) 형식
- 도표(Markdown table)는 섹션 3에서 반드시 2개 이상 포함
- 언어: 한국어 (기술 용어 영문 병기)
- 분량 부족 금지 — 각 소절 최소 600자, 전체 보고서 최소 3,000자
"""

# Section-specific search queries
_SECTION_QUERIES = {
    "background": [
        "AI memory bandwidth bottleneck HBM demand 2025",
        "data center memory market growth semiconductor",
    ],
    "technology_hbm4": [
        "HBM4 development status Samsung Micron 2025",
        "HBM4 product roadmap mass production",
        "HBM4 bandwidth stacking technology challenge",
    ],
    "technology_pim": [
        "PIM processing-in-memory product Samsung SK Hynix development",
        "PIM commercialization AI inference memory compute",
        "PIM hiring investment R&D signal 2024 2025",
    ],
    "technology_cxl": [
        "CXL memory pooling Intel AMD adoption data center",
        "CXL 3.0 standard memory interconnect roadmap",
        "CXL commercialization challenge bottleneck 2025",
    ],
    "competitor": [
        "Samsung HBM4 PIM CXL R&D investment strategy",
        "Micron HBM memory competitive strategy 2025",
        "Intel AMD CXL memory investment hiring",
    ],
    "future": [
        "next generation memory technology beyond HBM neuromorphic PIM",
        "CXL future standard 3D stacked memory concept",
        "semiconductor memory long term research emerging technology",
    ],
}


def _gather_section_evidence(technologies: list[str], competitors: list[str]) -> tuple[list[dict], dict[str, dict]]:
    """Run hybrid_search for each section and deduplicate by URL.

    Returns:
        sources: list of {id, title, url, date} — indexed starting at 1
        chunk_pool: dict[url, chunk] for deduplication
    """
    seen_urls: set[str] = set()
    ordered: list[dict] = []

    # Build queries: fixed section queries + per-technology + per-competitor
    all_queries: list[str] = []
    for qs in _SECTION_QUERIES.values():
        all_queries.extend(qs)
    for tech in technologies:
        all_queries.append(f"{tech} current development status 2024 2025")
    for company in competitors:
        all_queries.append(f"{company} semiconductor memory R&D strategy")

    for query in all_queries:
        for chunk in hybrid_search(query, top_k=5):
            url = chunk.get("meta", {}).get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            ordered.append(chunk)

    sources = [
        {
            "id": i + 1,
            "title": c["meta"].get("title", ""),
            "url": c["meta"].get("url", ""),
            "date": c["meta"].get("date", ""),
            "snippet": c.get("text", "")[:200],  # pass snippet so LLM can cite more specifically
        }
        for i, c in enumerate(ordered)
    ]
    return sources, {s["url"]: s for s in sources}


def report_agent(state: AgentState) -> dict:
    """T6: Draft full report using section-targeted hybrid_search evidence."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=16000)
    scope = state["scope"]
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])

    sources, _ = _gather_section_evidence(technologies, competitors)

    our_company = scope.get("our_company", "SK Hynix")
    context = (
        f"Our company (보고서 작성 주체): {our_company}\n"
        f"Competitors: {competitors}\n"
        f"Technologies: {technologies}\n\n"
        f"TRL Table:\n{json.dumps(state['trl_table'], ensure_ascii=False, indent=2)}\n\n"
        f"Threat Matrix:\n{json.dumps(state['threat_matrix'], ensure_ascii=False, indent=2)}\n\n"
        f"Available sources (use these for citations):\n{json.dumps(sources, ensure_ascii=False, indent=2)}"
    )

    response = llm.invoke([
        SystemMessage(content=REPORT_SYSTEM),
        HumanMessage(content=context),
    ])
    draft = response.content

    # Build reference list: in-text [N] → source entry via meta["url"]
    citation_ids = sorted(set(re.findall(r"\[(\d+)\]", draft)), key=int)
    id_map = {str(s["id"]): s for s in sources}
    today = date.today().isoformat()
    reference_list: list[ReferenceItem] = []
    for cid in citation_ids:
        src = id_map.get(cid, {})
        reference_list.append(
            ReferenceItem(
                citation_id=f"[{cid}]",
                url=src.get("url", ""),
                title=src.get("title", f"Source {cid}"),
                accessed_date=today,
            )
        )

    return {"draft_report": draft, "reference_list": reference_list}
