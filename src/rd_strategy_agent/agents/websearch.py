"""WebSearch Agent — Task T2 (parallel with Retrieve).

Sources:
- Tavily: multi-angle web search (news, blog, IR, hiring signals)
- OpenAlex: academic paper search (https://api.openalex.org/works)

Outcome: evidence_store populated with metadata-tagged snippets from both sources.
"""
from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import date

import aiohttp
from tavily import AsyncTavilyClient

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


def _tag_metadata(snippet: str, title: str, technologies: list[str], competitors: list[str]) -> tuple[list[str], list[str]]:
    text = (snippet + " " + title).lower()
    kws = [t for t in technologies if t.lower() in text]
    entities = [c for c in competitors if c.lower() in text]
    return kws, entities


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


async def _fetch_openalex_tech(
    session: aiohttp.ClientSession,
    tech: str,
    keywords: list[str],
    technologies: list[str],
    competitors: list[str],
    headers: dict,
) -> list[EvidenceItem]:
    query = tech
    related_kws = [kw for kw in keywords if tech.lower() in kw.lower()]
    if related_kws:
        query = related_kws[0]

    params = {
        "search": query,
        "sort": "publication_date:desc",
        "per_page": 10,
        "select": "id,doi,title,publication_date,abstract_inverted_index",
    }
    try:
        async with session.get(
            "https://api.openalex.org/works", headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            resp.raise_for_status()
            works = (await resp.json()).get("results", [])
    except Exception as e:
        print(f"[OpenAlex] query failed: {query!r} — {e}")
        return []

    results: list[EvidenceItem] = []
    for work in works:
        doi = work.get("doi") or ""
        url = doi if doi else work.get("id", "")
        if not url:
            continue
        title = work.get("title") or ""
        pub_date = work.get("publication_date") or ""
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        snippet = abstract if abstract else title
        kws, entities = _tag_metadata(snippet, title, technologies, competitors)
        results.append(
            EvidenceItem(
                url=url,
                title=title,
                date=pub_date,
                snippet=snippet,
                domain="openalex.org",
                keywords=kws,
                entities=entities,
            )
        )
    return results


async def _search_openalex_async(
    technologies: list[str], keywords: list[str], competitors: list[str]
) -> list[EvidenceItem]:
    api_key = os.environ.get("OPENALEX_API_KEY")
    headers = {"User-Agent": "rd-strategy-agent/0.1 (mailto:admin@example.com)"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_openalex_tech(session, tech, keywords, technologies, competitors, headers)
            for tech in technologies
        ]
        results_nested = await asyncio.gather(*tasks)

    return [ev for results in results_nested for ev in results]


async def _fetch_tavily(client: AsyncTavilyClient, query: str) -> list[dict]:
    try:
        results = await client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=False,
        )
        print(f"[WebSearch/Tavily] ✓ {query!r}")
        return results.get("results", [])
    except Exception as e:
        print(f"[WebSearch/Tavily] query failed: {query!r} — {e}")
        return []


async def _run_async(state: AgentState) -> dict:
    scope = state["scope"]
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    keywords = scope.get("keywords", [])

    seen_urls: set[str] = {ev["url"] for ev in state.get("evidence_store", [])}
    new_evidence: list[EvidenceItem] = []

    # --- Tavily (parallel) ---
    queries = _build_queries(technologies, competitors, keywords)
    print(f"[WebSearch] Firing {len(queries)} Tavily queries in parallel...")
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    raw_results = await asyncio.gather(*[_fetch_tavily(client, q) for q in queries])

    for results in raw_results:
        for r in results:
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

    # --- OpenAlex (parallel per tech) ---
    print("[WebSearch] Fetching OpenAlex papers in parallel...")
    openalex_results = await _search_openalex_async(technologies, keywords, competitors)
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
            print(f"[WebSearch] WARNING: domain '{dominant_domain}' covers {dominant_count/total:.0%} of results.")

    print(f"[WebSearch] Done — {total} evidence items collected.")
    return {"evidence_store": new_evidence}


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def websearch_agent(state: AgentState) -> dict:
    """T2: Multi-angle web search (Tavily) + academic paper search (OpenAlex)."""
    return asyncio.run(_run_async(state))
