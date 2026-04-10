"""
Academic Search Agent (T1 보조)
- 논문 검색: OpenAlex (1차) → Semantic Scholar (폴백)
- 특허 검색: Lens (1차) → PatentsView (폴백)
- 결과를 evidence_store에 source_type/doi/patent_number 메타데이터와 함께 저장
- PROJECT_PLAN 2.9 기반
"""

import os
import re
import requests
from datetime import date
from urllib.parse import urlparse

from agents.state import AgentState, EvidenceItem

# ── 기술 약어 → 풀네임 매핑 (오검색 방지) ────────────────────────
# "PIM" 단독 검색 시 Product Information Management로 오검색 됨
# "CXL" 단독 검색 시 Corneal Cross-Linking 의학 논문이 나옴
TECH_FULLNAME: dict[str, str] = {
    "PIM": "Processing-In-Memory semiconductor",
    "CXL": "Compute Express Link memory interconnect",
    "HBM4": "High Bandwidth Memory HBM4 DRAM",
}

# 반도체 도메인 관련성 판단 키워드 (최소 1개 이상 포함 필수)
SEMICONDUCTOR_KEYWORDS = {
    "semiconductor", "memory", "chip", "wafer", "dram", "nand", "flash",
    "hbm", "pim", "cxl", "compute", "processing", "bandwidth", "interconnect",
    "cache", "tsv", "packaging", "fabrication", "lithography", "transistor",
    "integrated circuit", "soc", "fpga", "gpu", "cpu", "trl", "ai accelerator",
    "near-memory", "in-memory", "compute express", "high bandwidth",
}

OPENALEX_BASE = "https://api.openalex.org/works"
LENS_PATENT_URL = "https://api.lens.org/patent/search"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PATENTSVIEW_URL = "https://api.patentsview.org/patents/query"


# ── OpenAlex 논문 검색 ────────────────────────────────────────────

def _search_openalex(query: str, per_page: int = 5) -> list[dict]:
    """OpenAlex API로 논문 검색. API 키 있으면 사용, 없으면 polite pool."""
    params = {
        "search": query,
        "per_page": per_page,
        "sort": "relevance_score:desc",
        "select": "id,doi,title,publication_date,authorships,abstract_inverted_index,primary_location",
    }
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    else:
        email = os.environ.get("OPENALEX_EMAIL", "")
        if email:
            params["mailto"] = email

    try:
        resp = requests.get(OPENALEX_BASE, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"[AcademicSearch] OpenAlex 오류: {e}")
        return []


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex inverted abstract index → plaintext"""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)[:500]


def _openalex_to_evidence(works: list[dict], scope: dict) -> list[EvidenceItem]:
    items = []
    for w in works:
        doi = w.get("doi", "") or ""
        title = w.get("title", "") or ""
        pub_date = w.get("publication_date", "") or date.today().isoformat()
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))

        url = doi if doi.startswith("http") else f"https://doi.org/{doi}" if doi else w.get("id", "")

        authors = w.get("authorships", [])
        institutions = []
        for a in authors[:3]:
            for inst in a.get("institutions", []):
                name = inst.get("display_name", "")
                if name:
                    institutions.append(name)

        kw, entities = _tag_item(abstract + " " + title, scope)
        items.append(EvidenceItem(
            url=url,
            title=title,
            date=str(pub_date)[:10],
            snippet=abstract,
            domain="openalex.org",
            keywords=kw,
            entities=entities,
            source_type="paper",
            doi=doi,
        ))
    return items


# ── Semantic Scholar 폴백 ─────────────────────────────────────────

def _search_semantic_scholar(query: str, limit: int = 5) -> list[dict]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,publicationDate,externalIds",
    }
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"[AcademicSearch] Semantic Scholar 폴백 오류: {e}")
        return []


def _semantic_scholar_to_evidence(papers: list[dict], scope: dict) -> list[EvidenceItem]:
    items = []
    for p in papers:
        doi = (p.get("externalIds") or {}).get("DOI", "")
        title = p.get("title", "") or ""
        abstract = (p.get("abstract") or "")[:500]
        pub_date = p.get("publicationDate", "") or date.today().isoformat()
        url = p.get("url", "") or (f"https://doi.org/{doi}" if doi else "")

        kw, entities = _tag_item(abstract + " " + title, scope)
        items.append(EvidenceItem(
            url=url,
            title=title,
            date=str(pub_date)[:10],
            snippet=abstract,
            domain="semanticscholar.org",
            keywords=kw,
            entities=entities,
            source_type="paper",
            doi=doi,
        ))
    return items


# ── Lens 특허 검색 ────────────────────────────────────────────────

def _search_lens_patents(query: str, size: int = 5) -> list[dict]:
    api_key = os.environ.get("LENS_API_KEY")
    if not api_key:
        return []

    body = {
        "query": {"match": {"full_text": query}},
        "size": size,
        "include": [
            "lens_id", "doc_key", "biblio.invention_title",
            "biblio.parties.applicants", "biblio.publication_reference",
            "abstract.text", "date_published",
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(LENS_PATENT_URL, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"[AcademicSearch] Lens 오류: {e}")
        return []


# ── PatentsView 폴백 (API 키 불필요, USPTO 데이터) ─────────────────

def _search_patentsview(query: str, size: int = 5) -> list[dict]:
    """USPTO PatentsView API — 키 없이 사용 가능한 무료 특허 검색"""
    # 쿼리에서 따옴표 제거 후 핵심 단어만 추출
    keywords = re.sub(r'["\']', '', query).strip()
    body = {
        "q": {"_text_any": {"patent_abstract": keywords}},
        "f": ["patent_number", "patent_title", "patent_abstract",
              "patent_date", "assignee_organization"],
        "o": {"per_page": size},
    }
    try:
        resp = requests.post(PATENTSVIEW_URL, json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("patents") or []
    except Exception as e:
        print(f"[AcademicSearch] PatentsView 오류: {e}")
        return []


def _patentsview_to_evidence(patents: list[dict], scope: dict) -> list[EvidenceItem]:
    items = []
    for p in patents:
        patent_number = p.get("patent_number", "")
        title = p.get("patent_title", "") or ""
        abstract = (p.get("patent_abstract") or "")[:500]
        pub_date = p.get("patent_date", "") or date.today().isoformat()
        assignee = p.get("assignee_organization", "") or ""
        url = f"https://patents.google.com/patent/US{patent_number}" if patent_number else ""

        kw, entities = _tag_item(abstract + " " + title, scope)
        items.append(EvidenceItem(
            url=url,
            title=title,
            date=str(pub_date)[:10],
            snippet=abstract,
            domain="patents.google.com",
            keywords=kw,
            entities=entities,
            source_type="patent",
            patent_number=patent_number,
            assignee=assignee,
        ))
    return items


def _lens_to_evidence(patents: list[dict], scope: dict) -> list[EvidenceItem]:
    items = []
    for p in patents:
        biblio = p.get("biblio", {})
        titles = biblio.get("invention_title", [])
        title = titles[0].get("text", "") if titles else ""

        abstracts = p.get("abstract", [])
        abstract = abstracts[0].get("text", "")[:500] if abstracts else ""

        pub_ref = biblio.get("publication_reference", {})
        patent_number = pub_ref.get("document_id", {}).get("doc_number", "")
        pub_date = p.get("date_published", "") or date.today().isoformat()
        lens_id = p.get("lens_id", "")
        url = f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else ""

        applicants = biblio.get("parties", {}).get("applicants", [])
        assignee = applicants[0].get("extracted_name", {}).get("value", "") if applicants else ""

        kw, entities = _tag_item(abstract + " " + title, scope)
        items.append(EvidenceItem(
            url=url,
            title=title,
            date=str(pub_date)[:10],
            snippet=abstract,
            domain="lens.org",
            keywords=kw,
            entities=entities,
            source_type="patent",
            patent_number=patent_number,
            assignee=assignee,
        ))
    return items


# ── 공통 헬퍼 ─────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """www., apps., m., emea., kr., us., eu. 등 서브도메인 변형 제거 → 중복 URL 판별에 사용"""
    try:
        p = urlparse(url)
        netloc = re.sub(r'^(www\.|apps\.|m\.|emea\.|kr\.|us\.|eu\.)+', '', p.netloc)
        path = p.path.rstrip('/')
        return f"{p.scheme}://{netloc}{path}"
    except Exception:
        return url


def _is_semiconductor_relevant(item: EvidenceItem) -> bool:
    """제목+스니펫에 반도체 도메인 키워드가 하나라도 있어야 통과"""
    text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
    return any(kw in text for kw in SEMICONDUCTOR_KEYWORDS)


def _tag_item(text: str, scope: dict) -> tuple[list[str], list[str]]:
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    lower = text.lower()
    kw = [t for t in technologies if t.lower() in lower]
    entities = [c for c in competitors if c.lower() in lower]
    return kw, entities


def _build_queries(scope: dict) -> list[tuple[str, str | None]]:
    """(query, query_company) 쌍 반환. 약어는 풀네임으로 치환해 오검색 방지."""
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    queries: list[tuple[str, str | None]] = []
    for tech in technologies:
        tech_q = TECH_FULLNAME.get(tech, f"{tech} semiconductor")
        queries.append((tech_q, None))
        for comp in competitors[:2]:
            queries.append((f"{comp} {tech_q}", comp))
    return queries


# ── 에이전트 ─────────────────────────────────────────────────────

def _tag_query_company(items: list[EvidenceItem], query_company: str | None) -> list[EvidenceItem]:
    """query_company가 있고 본문에 회사명이 없는 경우에만 query_company 태그 주입"""
    if not query_company:
        return items
    for item in items:
        text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
        if query_company.lower() not in text:
            item["query_company"] = query_company
    return items


def academic_search_agent(state: AgentState) -> dict:
    """논문(OpenAlex) + 특허(Lens) 검색 → evidence_store에 추가"""
    scope = state.get("scope", {})
    queries = _build_queries(scope)
    # URL 정규화 기반 중복 체크 (www. vs apps. 서브도메인 변형 통합)
    seen_norm = {_normalize_url(item["url"]) for item in state.get("evidence_store", [])}
    new_evidence: list[EvidenceItem] = []
    filtered_out = 0

    # 논문 검색
    print(f"[AcademicSearch] 논문 검색 중 ({len(queries)}개 쿼리, OpenAlex)...")
    paper_count = 0
    for query, query_company in queries:
        works = _search_openalex(query, per_page=3)
        if not works:
            works_fallback = _search_semantic_scholar(query, limit=3)
            items = _semantic_scholar_to_evidence(works_fallback, scope)
        else:
            items = _openalex_to_evidence(works, scope)

        items = _tag_query_company(items, query_company)
        for item in items:
            if not item["url"]:
                continue
            norm = _normalize_url(item["url"])
            if norm in seen_norm:
                continue
            if not _is_semiconductor_relevant(item):
                filtered_out += 1
                continue
            seen_norm.add(norm)
            new_evidence.append(item)
            paper_count += 1

    print(f"[AcademicSearch] 논문 {paper_count}건 수집")

    # 특허 검색 (Lens 1차 → PatentsView 폴백)
    has_lens_key = bool(os.environ.get("LENS_API_KEY"))
    source_label = "Lens → PatentsView 폴백" if not has_lens_key else "Lens"
    print(f"[AcademicSearch] 특허 검색 중 ({len(queries)}개 쿼리, {source_label})...")
    patent_count = 0
    for query, query_company in queries:
        patents = _search_lens_patents(query, size=3)
        if patents:
            items = _lens_to_evidence(patents, scope)
        else:
            # Lens 결과 없음(키 없거나 실패) → PatentsView 폴백
            pv_patents = _search_patentsview(query, size=3)
            items = _patentsview_to_evidence(pv_patents, scope)

        items = _tag_query_company(items, query_company)
        for item in items:
            if not item["url"]:
                continue
            norm = _normalize_url(item["url"])
            if norm in seen_norm:
                continue
            if not _is_semiconductor_relevant(item):
                filtered_out += 1
                continue
            seen_norm.add(norm)
            new_evidence.append(item)
            patent_count += 1

    print(f"[AcademicSearch] 특허 {patent_count}건 수집")
    if filtered_out:
        print(f"[AcademicSearch] 반도체 무관 노이즈 {filtered_out}건 필터링됨")
    print(f"[AcademicSearch] 총 신규 증거 {len(new_evidence)}건 추가")
    return {"evidence_store": new_evidence}
