"""
WebSearch Agent (T1, 병렬)
- Exa API로 다각도 쿼리 실행 (확증 편향 완화)
- 5가지 쿼리 유형: 현황 / 한계·비판 / 경쟁·대안 / 채용·투자 신호 / 반증
- 단일 도메인 40% 초과 시 경고
- 결과를 evidence_store 포맷으로 반환 → Supervisor에 보고
"""

import os
import re
from collections import Counter
from datetime import date
from urllib.parse import urlparse

from exa_py import Exa

from agents.state import AgentState, EvidenceItem


def _normalize_url(url: str) -> str:
    """www., apps., m., emea. 등 서브도메인 변형 제거 → 중복 URL 판별에 사용"""
    try:
        p = urlparse(url)
        netloc = re.sub(r'^(www\.|apps\.|m\.|emea\.|kr\.|us\.|eu\.)+', '', p.netloc)
        path = p.path.rstrip('/')
        return f"{p.scheme}://{netloc}{path}"
    except Exception:
        return url


# 반도체 도메인 관련성 키워드 (academic_search.py와 동일 기준)
_SEMICONDUCTOR_KEYWORDS = {
    "semiconductor", "memory", "chip", "wafer", "dram", "nand", "flash",
    "hbm", "pim", "cxl", "compute", "processing", "bandwidth", "interconnect",
    "cache", "tsv", "packaging", "fabrication", "lithography", "transistor",
    "integrated circuit", "soc", "fpga", "gpu", "cpu", "trl", "ai accelerator",
    "near-memory", "in-memory", "compute express", "high bandwidth",
}

# 노이즈 도메인 블랙리스트 (반도체 키워드와 무관한 동명 서비스만 등록)
# 채용공고(linkedin, talentify)·투자뉴스(globenewswire)는 TRL 신호로 유효 → 키워드 필터에 위임
_NOISE_DOMAINS = {
    "inriver.com",   # Marketing SaaS — PIM = Product Information Management
    "pimic.ai",      # Product Information Management SaaS — PIM 약어 오검색 유발
}

# 제목에 이 문자열이 있으면 반도체 키워드가 있어도 거부 (약어 오검색 방지)
# 예: "Product Information Management" → PIM 약어로 인해 반도체 키워드 매칭
_NOISE_TITLE_PATTERNS = [
    "product information management",   # PIM 약어 오검색
    "patient information",              # PIM 의료 약어
    "performance information management",
    "personal information management",
]

# 최소 허용 발행 연도 — 이 연도보다 오래된 기사는 기본적으로 필터링
# 환경변수로 조정 가능: EXA_MIN_YEAR=2022
_MIN_YEAR = int(os.environ.get("EXA_MIN_YEAR", "2023"))


def _is_semiconductor_relevant(snippet: str, title: str, url: str) -> bool:
    """반도체 도메인 관련성 판별:
    1. 블랙리스트 도메인이면 즉시 거부
    2. 제목에 노이즈 패턴(PIM 약어 오검색 등)이 있으면 거부
    3. 제목+스니펫에 반도체 키워드가 하나라도 있어야 통과
    """
    domain = url.split("/")[2] if url.startswith("http") else ""
    bare_domain = re.sub(r'^(www\.|apps\.|m\.|emea\.|kr\.|us\.|eu\.)+', '', domain)
    if bare_domain in _NOISE_DOMAINS:
        return False
    title_lower = title.lower()
    if any(pattern in title_lower for pattern in _NOISE_TITLE_PATTERNS):
        return False
    text = (title + " " + snippet).lower()
    return any(kw in text for kw in _SEMICONDUCTOR_KEYWORDS)


NUM_RESULTS = int(os.environ.get("EXA_NUM_RESULTS", 5))

QUERY_TEMPLATES = {
    "현황": '"{tech}" latest development {year}',
    "한계·비판": '"{tech}" limitations challenges problems',
    "경쟁·대안": '"{tech}" vs alternative comparison',
    "채용·투자 신호": '"{company}" "{tech}" hiring investment R&D',
    "반증": '"{tech}" failed abandoned setback',
    "스타트업": '"{tech}" startup emerging company funding venture {year}',
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

    # 자사 현황 쿼리
    self_company = scope.get("self_company", "SK Hynix")
    for tech in technologies:
        queries.append((f'"{self_company}" "{tech}" mass production shipment {year}', None))
        queries.append((f'"{self_company}" "{tech}" technology status TRL readiness', None))

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

    seen_norm = {_normalize_url(item["url"]) for item in state.get("evidence_store", [])}
    new_evidence: list[EvidenceItem] = []
    filtered_out = 0

    # Exa start_published_date 파라미터: _MIN_YEAR년 1월 1일 이후만 수집
    exa_date_filter = f"{_MIN_YEAR}-01-01"

    print(f"[WebSearch] {len(queries)}개 쿼리 실행 중 (Exa, {_MIN_YEAR}년~ 필터)...")
    filtered_old = 0
    for query, query_company in queries:
        try:
            results = client.search(
                query,
                num_results=NUM_RESULTS,
                start_published_date=exa_date_filter,
                contents={"highlights": {"max_characters": 500}},
            )
            for r in results.results:
                url = r.url or ""
                norm = _normalize_url(url)
                if norm in seen_norm:
                    continue

                snippet = " ".join(r.highlights) if r.highlights else ""
                snippet = snippet[:500]
                title = r.title or ""

                # 반도체 관련성 필터 (채용공고·무관 서비스·의학 도메인 차단)
                if not _is_semiconductor_relevant(snippet, title, url):
                    filtered_out += 1
                    continue

                seen_norm.add(norm)
                pub_date = r.published_date or date.today().isoformat()

                # 날짜 파싱 후 구형 기사 이중 체크 (API 필터가 누락할 수 있음)
                try:
                    pub_year = int(str(pub_date)[:4])
                    if pub_year < _MIN_YEAR:
                        filtered_old += 1
                        continue
                except (ValueError, TypeError):
                    pass  # 날짜 파싱 실패 시 통과

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

    if filtered_out:
        print(f"[WebSearch] 반도체 무관 노이즈 {filtered_out}건 필터링됨")
    if filtered_old:
        print(f"[WebSearch] {_MIN_YEAR}년 이전 구형 기사 {filtered_old}건 필터링됨")
    print(f"[WebSearch] 신규 증거 {len(new_evidence)}건 추가")
    return {"evidence_store": new_evidence}
