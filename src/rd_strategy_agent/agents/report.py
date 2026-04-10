"""Report Agent — Tasks T6, T7.

T6 Outcome: Draft report with required 4 sections + SUMMARY + REFERENCE.
T7 Outcome: REFERENCE ↔ in-text citation consistency verified, SUMMARY compressed.
"""
from __future__ import annotations

import re
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rd_strategy_agent.state import AgentState, ReferenceItem

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


def report_agent(state: AgentState) -> dict:
    """T6: Draft full report from analysis outputs."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=8000)

    import json
    evidence_urls = [
        {"id": i + 1, "title": ev["title"], "url": ev["url"], "date": ev["date"]}
        for i, ev in enumerate(state["evidence_store"][:40])
    ]
    context = (
        f"Scope: {json.dumps(state['scope'], ensure_ascii=False)}\n\n"
        f"TRL Table:\n{json.dumps(state['trl_table'], ensure_ascii=False, indent=2)}\n\n"
        f"Threat Matrix:\n{json.dumps(state['threat_matrix'], ensure_ascii=False, indent=2)}\n\n"
        f"Available sources (use these for citations):\n{json.dumps(evidence_urls, ensure_ascii=False, indent=2)}"
    )

    response = llm.invoke([
        SystemMessage(content=REPORT_SYSTEM),
        HumanMessage(content=context),
    ])
    draft = response.content
    if "## REFERENCE" not in draft:
        draft = draft.rstrip() + "\n\n## REFERENCE\n"

    # Build reference list from in-text citations
    citation_ids = sorted(set(re.findall(r"\[(\d+)\]", draft)), key=int)
    id_map = {str(e["id"]): e for e in evidence_urls}
    today = date.today().isoformat()
    reference_list: list[ReferenceItem] = []
    for cid in citation_ids:
        src = id_map.get(cid, {})
        reference_list.append(
            ReferenceItem(
                citation_id=f"[{cid}]",
                url=src.get("url", ""),
                title=src.get("title", "[매핑 불가]"),
                accessed_date=today,
            )
        )

    return {"draft_report": draft, "reference_list": reference_list}
