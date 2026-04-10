"""
WebSearch Agent (T2, 병렬)
- Tavily API로 다각도 쿼리 실행 (확증 편향 완화)
- 5가지 쿼리 유형: 현황 / 한계·비판 / 경쟁·대안 / 채용·투자 신호 / 반증
- 단일 도메인 40% 초과 시 경고
- 결과를 evidence_store 포맷으로 반환 → Supervisor에 보고
"""

import os
from collections import Counter
from datetime import date

from tavily import TavilyClient

from agents.state import AgentState, EvidenceItem


TOP_K = int(os.environ.get("TOP_K", 5))

QUERY_TEMPLATES = {
    "현황": '"{tech}" latest development {year}',
    "한계·비판": '"{tech}" limitations challenges problems',
    "경쟁·대안": '"{tech}" vs alternative comparison',
    "채용·투자 신호": '"{company}" "{tech}" hiring investment R&D',
    "반증": '"{tech}" failed abandoned setback',
}

TRUSTED_DOMAINS = {
    "ieee.org", "nature.com", "arxiv.org", "acm.org",
    "anandtech.com", "semianalysis.com", "tomshardware.com",
    "businesswire.com", "prnewswire.com", "sec.gov",
}


def _build_queries(scope: dict, iteration_count: int) -> list[str]:
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    year = date.today().year
    queries = []

    for tech in technologies:
        for qtype, template in QUERY_TEMPLATES.items():
            if "{company}" in template:
                for comp in competitors[:2]:  # 대표 경쟁사 2곳만
                    queries.append(template.format(tech=tech, company=comp, year=year))
            else:
                queries.append(template.format(tech=tech, year=year))

    # 재시도 시 경쟁사 쿼리 확장
    if iteration_count > 0:
        for comp in competitors:
            for tech in technologies:
                queries.append(f'"{comp}" "{tech}" TRL technology roadmap {year}')

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
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("[WebSearch] TAVILY_API_KEY 없음 — 스킵")
        return {}

    client = TavilyClient(api_key=api_key)
    scope = state.get("scope", {})
    iteration_count = state.get("iteration_count", 0)
    queries = _build_queries(scope, iteration_count)

    seen_urls = {item["url"] for item in state.get("evidence_store", [])}
    new_evidence: list[EvidenceItem] = []

    print(f"[WebSearch] {len(queries)}개 쿼리 실행 중...")
    for query in queries:
        try:
            results = client.search(
                query=query,
                max_results=TOP_K,
                include_raw_content=False,
            )
            for r in results.get("results", []):
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                snippet = r.get("content", "")[:500]
                title = r.get("title", "")
                pub_date = r.get("published_date", date.today().isoformat()) or date.today().isoformat()
                domain = url.split("/")[2] if url.startswith("http") else "unknown"

                kw, entities = _tag_item(snippet, title, scope)
                new_evidence.append(EvidenceItem(
                    url=url,
                    title=title,
                    date=pub_date[:10],
                    snippet=snippet,
                    domain=domain,
                    keywords=kw,
                    entities=entities,
                ))
        except Exception as e:
            print(f"[WebSearch] 쿼리 실패 ({query[:40]}...): {e}")

    # 출처 다양성 체크 (전체 evidence_store 기준)
    all_evidence = state.get("evidence_store", []) + new_evidence
    _check_domain_diversity(all_evidence)

    print(f"[WebSearch] 신규 증거 {len(new_evidence)}건 추가")
    return {"evidence_store": new_evidence}
