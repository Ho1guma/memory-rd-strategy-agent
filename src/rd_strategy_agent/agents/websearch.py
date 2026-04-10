"""WebSearch Agent — Task T2 (parallel with Retrieve).

Uses Tavily to fetch evidence with multi-angle queries to reduce confirmation bias.
Outcome: evidence_store populated with metadata-tagged snippets.
"""
from __future__ import annotations

import os
from datetime import date

from tavily import TavilyClient

from rd_strategy_agent.state import AgentState, EvidenceItem

# Query templates per angle (SC2.8 — confirmation bias mitigation)
QUERY_TEMPLATES = [
    '"{tech}" latest development {year}',
    '"{tech}" limitations challenges',
    '"{tech}" vs alternative comparison',
    '"{company}" "{tech}" hiring investment',
    '"{tech}" failed abandoned setback',
]


def _build_queries(technologies: list[str], competitors: list[str], keywords: list[str]) -> list[str]:
    year = date.today().year
    queries: list[str] = []
    for tech in technologies:
        for tmpl in QUERY_TEMPLATES:
            if "{company}" in tmpl:
                for comp in competitors[:3]:  # limit to top 3 to control API cost
                    queries.append(tmpl.format(tech=tech, company=comp, year=year))
            else:
                queries.append(tmpl.format(tech=tech, year=year))
    # Add keyword-based queries
    for kw in keywords[:5]:
        queries.append(kw)
    return queries


def _tag_metadata(snippet: str, title: str, technologies: list[str], competitors: list[str]) -> tuple[list[str], list[str]]:
    text = (snippet + " " + title).lower()
    kws = [t for t in technologies if t.lower() in text]
    entities = [c for c in competitors if c.lower() in text]
    return kws, entities


def websearch_agent(state: AgentState) -> dict:
    """T2: Multi-angle web search via Tavily."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    scope = state["scope"]
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    keywords = scope.get("keywords", [])

    queries = _build_queries(technologies, competitors, keywords)
    new_evidence: list[EvidenceItem] = []
    seen_urls: set[str] = {ev["url"] for ev in state.get("evidence_store", [])}

    for query in queries:
        try:
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_raw_content=False,
            )
            for r in results.get("results", []):
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                snippet = r.get("content", "")
                title = r.get("title", "")
                domain = url.split("/")[2] if url.startswith("http") else ""
                kws, entities = _tag_metadata(snippet, title, technologies, competitors)
                new_evidence.append(
                    EvidenceItem(
                        url=url,
                        title=title,
                        date=r.get("published_date", ""),
                        snippet=snippet,
                        domain=domain,
                        keywords=kws,
                        entities=entities,
                    )
                )
        except Exception as e:
            print(f"[WebSearch] query failed: {query!r} — {e}")

    # Source diversity check: flag if any domain > 40% of total
    total = len(new_evidence)
    if total > 0:
        from collections import Counter
        domain_counts = Counter(ev["domain"] for ev in new_evidence)
        dominant = domain_counts.most_common(1)[0]
        if dominant[1] / total > 0.40:
            print(f"[WebSearch] WARNING: domain {dominant[0]} covers {dominant[1]/total:.0%} of results.")

    return {"evidence_store": new_evidence}
