"""Analysis Agent — Tasks T4, T5.

T4 Outcome: TRL estimation table (per company × technology) with evidence_count ≥ 2.
T5 Outcome: Threat matrix (per company) with level + rationale.

Evidence is retrieved via hybrid_search (dense + BM25 + RRF) per (company, technology)
pair rather than passing the full evidence_store to the LLM.
"""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rd_strategy_agent.state import AgentState, TRLEntry, ThreatEntry
from rd_strategy_agent.agents.retrieve import hybrid_search

TRL_SYSTEM = """You are a semiconductor R&D intelligence analyst.
Given collected evidence snippets, estimate TRL (1–9) for each (company, technology) pair.

Rules:
- TRL 1–3: Use paper/patent/conference evidence.
- TRL 4–6: MUST label as "estimated". State indirect indicators (patent filing trends, hiring keywords, conference frequency). Do NOT assert exact TRL — give a range (e.g., "5-6").
- TRL 7–9: Require mass production, shipment, or official IR evidence.
- You MUST return one row for every (company, technology) pair in scope.
- Each entry requires evidence_count >= 2 citations. If evidence is insufficient, set {"trl_range": "정보 부족", "evidence_count": 0 or 1, "label": "insufficient", "sources": [] or partial}.

Output JSON array with this schema (no markdown fences):
[{"company": str, "technology": str, "trl_range": str, "evidence_count": int, "label": "confirmed"|"estimated"|"insufficient", "sources": [citation_id, ...]}]
"""

THREAT_SYSTEM = """You are a technology competitive strategy analyst.
Given the TRL table and threat level rules, assign a threat level to each competitor.

Threat rules:
- Use the provided scope threat_level_rules exactly.
- Base the company threat level on the highest valid TRL found across that company's technologies.
- Ignore rows with trl_range="정보 부족" when picking the base TRL.
- Upgrade one level if investment or hiring signals are present in evidence.
- Return exactly one row per competitor in scope.

Output JSON array (no markdown fences):
[{"company": str, "level": "low"|"medium"|"high", "rationale": str}]
"""


def _build_pair_evidence(companies: list[str], technologies: list[str]) -> str:
    """Run hybrid_search per (company, technology) pair and format as evidence block."""
    blocks: list[str] = []
    for company in companies:
        for tech in technologies:
            query = f"{company} {tech} TRL development production status"
            chunks = hybrid_search(query, top_k=10)
            if not chunks:
                continue
            lines = [f"### Evidence: {company} × {tech}"]
            for chunk in chunks:
                meta = chunk.get("meta", {})
                title = meta.get("title", "")
                url = meta.get("url", "")
                date = meta.get("date", "")
                text = chunk["text"][:300]
                lines.append(f"- [{title}]({url}) ({date})\n  {text}")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def analysis_agent(state: AgentState) -> dict:
    """T4 + T5: Estimate TRL and threat level for each competitor."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    scope = state["scope"]
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])

    evidence_text = _build_pair_evidence(competitors, technologies)

    our_company = scope.get("our_company", "SK Hynix")
    context = (
        f"Our company (분석 주체): {our_company}\n"
        f"Technologies: {technologies}\n"
        f"Competitors (경쟁사만 포함, 당사 제외): {competitors}\n"
        f"Threat level rules from scope.yaml: {scope.get('threat_level_rules', {})}\n\n"
        f"Evidence (retrieved per company × technology):\n{evidence_text}"
    )

    # T4 — TRL table
    trl_response = llm.invoke([
        SystemMessage(content=TRL_SYSTEM),
        HumanMessage(content=context),
    ])
    trl_raw = trl_response.content.strip().lstrip("```json").lstrip("```").rstrip("```")
    trl_table: list[TRLEntry] = json.loads(trl_raw)
    trl_table = _normalize_trl_table(trl_table, scope)

    # T5 — Threat matrix
    threat_context = context + f"\n\nTRL Table:\n{json.dumps(trl_table, ensure_ascii=False, indent=2)}"
    threat_response = llm.invoke([
        SystemMessage(content=THREAT_SYSTEM),
        HumanMessage(content=threat_context),
    ])
    threat_raw = threat_response.content.strip().lstrip("```json").lstrip("```").rstrip("```")
    threat_matrix: list[ThreatEntry] = json.loads(threat_raw)
    threat_matrix = _normalize_threat_matrix(threat_matrix, scope)

    return {"trl_table": trl_table, "threat_matrix": threat_matrix}


def _normalize_trl_table(trl_table: list[TRLEntry], scope: dict) -> list[TRLEntry]:
    by_pair = {(row["company"], row["technology"]): row for row in trl_table}
    normalized: list[TRLEntry] = []
    for company in scope.get("competitors", []):
        for technology in scope.get("technologies", []):
            row = by_pair.get((company, technology))
            if row:
                normalized.append(row)
                continue
            normalized.append(
                TRLEntry(
                    company=company,
                    technology=technology,
                    trl_range="정보 부족",
                    evidence_count=0,
                    label="insufficient",
                    sources=[],
                )
            )
    return normalized


def _normalize_threat_matrix(threat_matrix: list[ThreatEntry], scope: dict) -> list[ThreatEntry]:
    by_company = {row["company"]: row for row in threat_matrix}
    normalized: list[ThreatEntry] = []
    for company in scope.get("competitors", []):
        row = by_company.get(company)
        if row:
            normalized.append(row)
            continue
        normalized.append(
            ThreatEntry(
                company=company,
                level="low",
                rationale="자동 보정: 분석 결과 누락",
            )
        )
    return normalized
