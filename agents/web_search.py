"""
WebSearch Agent (T1, 병렬)
- Exa API로 다각도 쿼리 실행 (확증 편향 완화)
- 5가지 쿼리 유형: 현황 / 한계·비판 / 경쟁·대안 / 채용·투자 신호 / 반증
- 단일 도메인 40% 초과 시 경고
- 결과를 evidence_store 포맷으로 반환 → Supervisor에 보고
"""

import os
from collections import Counter
from datetime import date

from exa_py import Exa

from agents.state import AgentState, EvidenceItem


NUM_RESULTS = int(os.environ.get("EXA_NUM_RESULTS", 5))

QUERY_TEMPLATES = {
    "현황": '"{tech}" latest development {year}',
    "한계·비판": '"{tech}" limitations challenges problems',
    "경쟁·대안": '"{tech}" vs alternative comparison',
    "채용·투자 신호": '"{company}" "{tech}" hiring investment R&D',
    "반증": '"{tech}" failed abandoned setback',
}


def _build_queries(scope: dict, iteration_count: int) -> list[tuple[str, str | None]]:
    """(query, query_company) 쌍 반환. 회사 무관 쿼리는 query_company=None"""
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    year = date.today().year
    queries: list[tuple[str, str | None]] = []

    for tech in technologies:
        for qtype, template in QUERY_TEMPLATES.items():
            if "{company}" in template:
                for comp in competitors[:2]:
                    queries.append((template.format(tech=tech, company=comp, year=year), comp))
            else:
                queries.append((template.format(tech=tech, year=year), None))

    # SK Hynix 자사 현황 쿼리
    for tech in technologies:
        queries.append((f'"SK Hynix" "{tech}" mass production shipment {year}', None))
        queries.append((f'"SK Hynix" "{tech}" technology status TRL readiness', None))

    if iteration_count > 0:
        for comp in competitors:
            for tech in technologies:
                queries.append((f'"{comp}" "{tech}" TRL technology roadmap {year}', comp))

    return queries


def _tag_item(snippet: str, title: str, scope: dict) -> tuple[list[str], list[str]]:
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    text = (snippet + " " + title).lower()
    kw = [t for t in technologies if t.lower() in text]
    entities = [c for c in competitors if c.lower() in text]
    return kw, entities


def _check_domain_diversity(evidence_store: list[EvidenceItem]) -> None:
    domains = [item["domain"] for item in evidence_store]
    total = len(domains)
    if total == 0:
        return
    counts = Counter(domains)
    for domain, count in counts.most_common(1):
        ratio = count / total
        if ratio > 0.4:
            print(f"[WebSearch] ⚠️ 출처 다양성 경고: '{domain}' {ratio:.0%} 집중 → Supervisor 재시도 트리거 가능")


def web_search_agent(state: AgentState) -> dict:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        print("[WebSearch] EXA_API_KEY 없음 — 스킵")
        return {}

    client = Exa(api_key=api_key)
    scope = state.get("scope", {})
    iteration_count = state.get("iteration_count", 0)
    queries = _build_queries(scope, iteration_count)

    seen_urls = {item["url"] for item in state.get("evidence_store", [])}
    new_evidence: list[EvidenceItem] = []

    print(f"[WebSearch] {len(queries)}개 쿼리 실행 중 (Exa)...")
    for query, query_company in queries:
        try:
            results = client.search(
                query,
                num_results=NUM_RESULTS,
                contents={"highlights": {"max_characters": 500}},
            )
            for r in results.results:
                url = r.url or ""
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                snippet = " ".join(r.highlights) if r.highlights else ""
                snippet = snippet[:500]
                title = r.title or ""
                pub_date = r.published_date or date.today().isoformat()
                domain = url.split("/")[2] if url.startswith("http") else "unknown"

                kw, entities = _tag_item(snippet, title, scope)
                item = EvidenceItem(
                    url=url,
                    title=title,
                    date=str(pub_date)[:10],
                    snippet=snippet,
                    domain=domain,
                    keywords=kw,
                    entities=entities,
                    source_type="web",
                )
                # 본문에 회사명이 없는데 회사 쿼리로 찾은 결과면 query_company 태그
                if query_company:
                    text = (title + " " + snippet).lower()
                    if query_company.lower() not in text:
                        item["query_company"] = query_company
                new_evidence.append(item)
        except Exception as e:
            print(f"[WebSearch] 쿼리 실패 ({query[:40]}...): {e}")

    all_evidence = state.get("evidence_store", []) + new_evidence
    _check_domain_diversity(all_evidence)

    print(f"[WebSearch] 신규 증거 {len(new_evidence)}건 추가")
    return {"evidence_store": new_evidence}
