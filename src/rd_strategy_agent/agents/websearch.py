"""WebSearch Agent — Task T2 (parallel with Retrieve).

Sources:
- Exa: multi-angle web search (news, blog, IR, hiring signals)
- OpenAlex: academic paper search (https://api.openalex.org/works)

Outcome: evidence_store populated with metadata-tagged snippets from both sources.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import date

import requests
from exa_py import Exa

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
    for kw in keywords[:5]:
        queries.append(kw)
    return queries


def _tag_metadata(
    snippet: str, title: str, technologies: list[str], competitors: list[str]
) -> tuple[list[str], list[str], str]:
    text = (snippet + " " + title).lower()
    kws = [t for t in technologies if t.lower() in text]
    entities = [c for c in competitors if c.lower() in text]
    tagging_status = "ok" if kws or entities else "tagging_unavailable"
    return kws, entities, tagging_status


def _get_result_field(result: object, field: str, default: str = "") -> str:
    if isinstance(result, dict):
        value = result.get(field, default)
    else:
        value = getattr(result, field, default)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    return value or default


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Convert OpenAlex abstract_inverted_index to plain text.

    Format: {"word": [pos1, pos2, ...], ...}
    """
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def _search_openalex(technologies: list[str], keywords: list[str], competitors: list[str]) -> list[EvidenceItem]:
    """Fetch top-10 latest papers per technology from OpenAlex."""
    api_key = os.environ.get("OPENALEX_API_KEY")
    headers = {"User-Agent": "rd-strategy-agent/0.1 (mailto:admin@example.com)"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results: list[EvidenceItem] = []

    for tech in technologies:
        query = tech
        # Augment with relevant keywords if available
        related_kws = [kw for kw in keywords if tech.lower() in kw.lower()]
        if related_kws:
            query = related_kws[0]

        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                headers=headers,
                params={
                    "search": query,
                    "sort": "publication_date:desc",
                    "per_page": 10,
                    "select": "id,doi,title,publication_date,abstract_inverted_index",
                },
                timeout=15,
            )
            resp.raise_for_status()
            works = resp.json().get("results", [])
        except Exception as e:
            print(f"[OpenAlex] query failed: {query!r} — {e}")
            continue

        for work in works:
            doi = work.get("doi") or ""
            url = doi if doi else work.get("id", "")
            if not url:
                continue
            title = work.get("title") or ""
            pub_date = work.get("publication_date") or ""
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            snippet = abstract if abstract else title
            kws, entities, tagging_status = _tag_metadata(snippet, title, technologies, competitors)
            results.append(
                EvidenceItem(
                    url=url,
                    title=title,
                    date=pub_date,
                    snippet=snippet,
                    domain="openalex.org",
                    keywords=kws,
                    entities=entities,
                    tagging_status=tagging_status,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def websearch_agent(state: AgentState) -> dict:
    """T2: Multi-angle web search (Exa) + academic paper search (OpenAlex)."""
    scope = state["scope"]
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    keywords = scope.get("keywords", [])

    new_evidence: list[EvidenceItem] = []
    seen_urls: set[str] = {ev["url"] for ev in state.get("evidence_store", [])}

    # --- Exa ---
    client = Exa(api_key=os.environ["EXA_API_KEY"])
    queries = _build_queries(technologies, competitors, keywords)
    for query in queries:
        try:
            results = client.search_and_contents(
                query,
                type="auto",
                text={"max_characters": 4000},
                num_results=5,
            )
            for r in getattr(results, "results", []):
                url = _get_result_field(r, "url")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                snippet = _get_result_field(r, "text") or _get_result_field(r, "highlights")
                title = _get_result_field(r, "title")
                domain = url.split("/")[2] if url.startswith("http") else ""
                kws, entities, tagging_status = _tag_metadata(snippet, title, technologies, competitors)
                new_evidence.append(
                    EvidenceItem(
                        url=url,
                        title=title,
                        date=_get_result_field(r, "published_date"),
                        snippet=snippet,
                        domain=domain,
                        keywords=kws,
                        entities=entities,
                        tagging_status=tagging_status,
                    )
                )
        except Exception as e:
            print(f"[WebSearch/Exa] query failed: {query!r} — {e}")

    # --- OpenAlex ---
    openalex_results = _search_openalex(technologies, keywords, competitors)
    for ev in openalex_results:
        if ev["url"] in seen_urls:
            continue
        seen_urls.add(ev["url"])
        new_evidence.append(ev)

    # --- Source diversity check ---
    total = len(new_evidence)
    if total > 0:
        domain_counts = Counter(ev["domain"] for ev in new_evidence)
        dominant_domain, dominant_count = domain_counts.most_common(1)[0]
        if dominant_count / total > 0.40:
            # openalex.org concentration is expected for paper-heavy topics — WARNING only
            print(f"[WebSearch] WARNING: domain '{dominant_domain}' covers {dominant_count/total:.0%} of results.")

    return {"evidence_store": new_evidence}
