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

REPORT_SYSTEM = """You are a technical strategy report writer for a semiconductor R&D team.

Write a Korean-language R&D strategy analysis report in Markdown with this exact structure:

## SUMMARY
(½ page max — key findings and recommendations)

## 1. 분석 배경
(Why these technologies matter now)

## 2. 분석 대상 기술 현황
(HBM4, PIM, CXL — current state, direction, bottlenecks per technology)

## 3. 경쟁사 동향 분석
(Competitor strategies, TRL summary table, threat matrix)

## 4. 전략적 시사점
(R&D priority recommendations — short term / medium term)

## REFERENCE
(Numbered list: [N] Title. URL. Accessed YYYY-MM-DD.)

Rules:
- Use [N] citation format inline (e.g., Samsung has achieved TRL 7 [3]).
- TRL 4–6 entries MUST be labeled "(추정)" and include indirect indicators.
- Do NOT fabricate URLs — only use sources provided.
- Language: Korean (technical terms in English where standard).
"""

# Section-specific search queries
_SECTION_QUERIES = {
    "background": ["semiconductor memory market trend 2024 2025", "AI memory bandwidth bottleneck"],
    "technology": ["HBM4 current status development", "PIM architecture deployment", "CXL memory pooling interconnect"],
    "competitor": ["Samsung SK Hynix Micron HBM competitive strategy", "Intel AMD CXL adoption"],
    "strategy": ["memory semiconductor R&D priority investment", "next generation memory roadmap"],
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
        for chunk in hybrid_search(query, top_k=10):
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
        }
        for i, c in enumerate(ordered)
    ]
    return sources, {s["url"]: s for s in sources}


def report_agent(state: AgentState) -> dict:
    """T6: Draft full report using section-targeted hybrid_search evidence."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=8000)
    scope = state["scope"]
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])

    sources, _ = _gather_section_evidence(technologies, competitors)

    context = (
        f"Scope: {json.dumps(scope, ensure_ascii=False)}\n\n"
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
