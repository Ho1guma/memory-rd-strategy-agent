"""
Academic Search Agent (T1 보조)
- 논문 검색: OpenAlex (1차) → Semantic Scholar (폴백)
- 특허 검색: Lens (1차) → PatentsView (폴백)
- 결과를 evidence_store에 source_type/doi/patent_number 메타데이터와 함께 저장
- PROJECT_PLAN 2.9 기반
"""

import os
import requests
from datetime import date

from agents.state import AgentState, EvidenceItem

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
        print("[AcademicSearch] LENS_API_KEY 없음 — 특허 검색 스킵")
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

def _tag_item(text: str, scope: dict) -> tuple[list[str], list[str]]:
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    lower = text.lower()
    kw = [t for t in technologies if t.lower() in lower]
    entities = [c for c in competitors if c.lower() in lower]
    return kw, entities


def _build_queries(scope: dict) -> list[tuple[str, str | None]]:
    """(query, query_company) 쌍 반환. 회사 무관 쿼리는 query_company=None"""
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    queries: list[tuple[str, str | None]] = []
    for tech in technologies:
        queries.append((f"{tech} semiconductor", None))
        for comp in competitors[:2]:
            queries.append((f"{comp} {tech}", comp))
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
    seen_urls = {item["url"] for item in state.get("evidence_store", [])}
    new_evidence: list[EvidenceItem] = []

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
            if item["url"] and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                new_evidence.append(item)
                paper_count += 1

    print(f"[AcademicSearch] 논문 {paper_count}건 수집")

    # 특허 검색
    print(f"[AcademicSearch] 특허 검색 중 ({len(queries)}개 쿼리, Lens)...")
    patent_count = 0
    for query, query_company in queries:
        patents = _search_lens_patents(query, size=3)
        items = _lens_to_evidence(patents, scope)
        items = _tag_query_company(items, query_company)
        for item in items:
            if item["url"] and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                new_evidence.append(item)
                patent_count += 1

    print(f"[AcademicSearch] 특허 {patent_count}건 수집")
    print(f"[AcademicSearch] 총 신규 증거 {len(new_evidence)}건 추가")
    return {"evidence_store": new_evidence}
