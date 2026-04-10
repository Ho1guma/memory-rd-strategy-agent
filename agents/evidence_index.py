"""
Evidence Index (벡터 DB 래퍼)

WebSearch + AcademicSearch 수집 완료 후 evidence_store를 FAISS로 인덱싱.
Analysis/Report 에이전트가 company×technology별 쿼리로 관련 증거 top-K만 조회.

- 임베딩: BAAI/bge-m3 (retrieve agent와 동일 모델, 모듈 레벨 캐싱)
- 저장: 프로세스 내 전역 변수 (FAISS 객체는 직렬화 불가 → TypedDict state 저장 불가)
- 검색 필터: 유사도 점수 임계값 + 반도체 도메인 관련성 필터 + 날짜 페널티
"""

import os
from datetime import date
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agents.state import EvidenceItem
from agents.embeddings import get_embeddings

# FAISS L2 거리 기준: 낮을수록 유사. bge-m3 정규화 벡터 기준 1.2 이상이면 관련성 낮음
# (0=동일, 2=완전 반대 방향). 환경변수로 조정 가능
SCORE_THRESHOLD = float(os.environ.get("EVIDENCE_SCORE_THRESHOLD", "1.2"))

# 날짜 페널티: 현재 연도 - 기사 연도가 클수록 페널티. 이 값을 초과하면 검색 결과에서 제외.
# 환경변수로 조정 가능: EVIDENCE_MAX_AGE_YEARS=3
_MAX_AGE_YEARS = int(os.environ.get("EVIDENCE_MAX_AGE_YEARS", "3"))
_CURRENT_YEAR = date.today().year


def _date_penalty(item: EvidenceItem) -> float:
    """기사 연도에 따른 L2 거리 페널티 반환.
    - 올해/작년: 0.0 (페널티 없음)
    - 2년 전: 0.1
    - 3년 전: 0.2
    - 4년 이상: 0.4 (거의 제외 수준)
    논문·특허는 페널티 절반 적용 (연구 성과는 발행 후에도 유효)
    """
    pub_date = item.get("date", "") or item.get("publication_date", "")
    try:
        pub_year = int(str(pub_date)[:4])
    except (ValueError, TypeError):
        return 0.0  # 날짜 불명확 → 페널티 없음

    age = _CURRENT_YEAR - pub_year
    if age <= 1:
        penalty = 0.0
    elif age == 2:
        penalty = 0.1
    elif age == 3:
        penalty = 0.2
    else:
        penalty = 0.4

    # 논문·특허는 절반 페널티
    if item.get("source_type") in ("paper", "patent"):
        penalty *= 0.5

    return penalty

# 반도체 도메인 관련성 키워드 (academic_search.py와 동일 기준)
_SEMICONDUCTOR_KEYWORDS = {
    "semiconductor", "memory", "chip", "wafer", "dram", "nand", "flash",
    "hbm", "pim", "cxl", "compute", "processing", "bandwidth", "interconnect",
    "cache", "tsv", "packaging", "fabrication", "lithography", "transistor",
    "integrated circuit", "soc", "fpga", "gpu", "cpu", "trl", "ai accelerator",
    "near-memory", "in-memory", "compute express", "high bandwidth",
}

# FAISS 인덱스 싱글턴 (임베딩 싱글턴은 agents.embeddings에서 관리)
_faiss_index: Optional[FAISS] = None
_evidence_map: dict[int, EvidenceItem] = {}  # FAISS doc id → EvidenceItem


def build_evidence_index(evidence_store: list[EvidenceItem]) -> bool:
    """
    evidence_store 전체를 FAISS로 인덱싱.
    성공 시 True, 실패 시 False 반환.
    """
    global _faiss_index, _evidence_map

    if not evidence_store:
        print("[EvidenceIndex] evidence_store 비어있음 — 인덱싱 스킵")
        return False

    print(f"[EvidenceIndex] {len(evidence_store)}건 인덱싱 중...")

    docs = []
    _evidence_map = {}
    for i, item in enumerate(evidence_store):
        text = f"{item.get('title', '')} {item.get('snippet', '')}"
        doc = Document(
            page_content=text[:1000],
            metadata={"evidence_idx": i},
        )
        docs.append(doc)
        _evidence_map[i] = item

    try:
        embeddings = get_embeddings()
        _faiss_index = FAISS.from_documents(docs, embeddings)
        print(f"[EvidenceIndex] 인덱싱 완료 ({len(docs)}건)")
        return True
    except Exception as e:
        print(f"[EvidenceIndex] 인덱싱 실패: {e}")
        return False


_NOISE_DOMAINS = {
    "inriver.com", "pimic.ai", "akeneo.com", "salsify.com",
    "struct.com", "gepard.io", "startwithdata.com", "startwithdata.co.uk",
    "stedger.com", "plytix.com", "catsy.com", "contentserv.com",
    "stibo.com", "riversand.com", "syndigo.com",
}

_NOISE_TITLE_PATTERNS = [
    "product information management", "patient information",
    "pim implementation", "pim solution", "pim software",
    "pim changelog", "pim scalab", "pim migration", "pim integration",
    "pim platform", "winter release", "spring release", "agentic ai product",
]


def _is_semiconductor_relevant(item: EvidenceItem) -> bool:
    """제목+스니펫에 반도체 도메인 키워드가 있고, 노이즈 도메인/제목이 아니어야 통과"""
    url = item.get("url", "")
    title = item.get("title", "")

    # 도메인 블랙리스트
    import re as _re
    domain = url.split("/")[2] if url.startswith("http") else ""
    bare = _re.sub(r'^(www\.|apps\.|m\.|emea\.|kr\.|us\.|eu\.)+', '', domain)
    if bare in _NOISE_DOMAINS:
        return False

    # 제목 노이즈 패턴
    title_lower = title.lower()
    if any(p in title_lower for p in _NOISE_TITLE_PATTERNS):
        return False

    text = (title + " " + item.get("snippet", "")).lower()
    return any(kw in text for kw in _SEMICONDUCTOR_KEYWORDS)


def query_evidence(query: str, k: int = 8, score_threshold: float | None = None) -> list[EvidenceItem]:
    """
    쿼리와 관련된 증거 top-k를 반환.
    - score_threshold: FAISS L2 거리 상한 (기본값: SCORE_THRESHOLD 환경변수)
      낮은 유사도(높은 거리) 항목을 제거해 노이즈 차단
    - 반도체 도메인 관련성 필터 적용
    - 인덱스 미구축 시 빈 리스트 반환.
    """
    if _faiss_index is None or not _evidence_map:
        return []

    threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD

    try:
        # k*2 후보를 뽑아 필터링 후 k개 반환
        fetch_k = min(k * 2, len(_evidence_map))
        results_with_scores = _faiss_index.similarity_search_with_score(query, k=fetch_k)

        evidence_items = []
        seen_idx = set()
        filtered_score = 0
        filtered_domain = 0

        filtered_old = 0
        for doc, score in results_with_scores:
            if len(evidence_items) >= k:
                break
            idx = doc.metadata.get("evidence_idx")
            if idx is None or idx in seen_idx or idx not in _evidence_map:
                continue
            item = _evidence_map[idx]

            # 날짜 페널티를 L2 거리에 가산해 구형 기사를 자연스럽게 후순위로
            penalty = _date_penalty(item)
            effective_score = score + penalty

            if effective_score > threshold:
                filtered_score += 1
                # 4년+ 구형 기사는 별도 카운트
                if penalty >= 0.4:
                    filtered_old += 1
                continue
            if not _is_semiconductor_relevant(item):
                filtered_domain += 1
                continue
            evidence_items.append(item)
            seen_idx.add(idx)

        if filtered_score or filtered_domain or filtered_old:
            print(
                f"[EvidenceIndex] 쿼리 필터: 점수초과 {filtered_score}건 (구형기사 {filtered_old}건 포함), "
                f"도메인무관 {filtered_domain}건 제거 → {len(evidence_items)}건 반환"
            )
        return evidence_items
    except Exception as e:
        print(f"[EvidenceIndex] 쿼리 실패 ({query[:40]}...): {e}")
        return []


def is_index_ready() -> bool:
    return _faiss_index is not None and len(_evidence_map) > 0


def get_evidence_by_idx(idx: int) -> Optional[EvidenceItem]:
    return _evidence_map.get(idx)


def get_all_evidence() -> list[EvidenceItem]:
    """인덱스 미구축 시 폴백용 전체 evidence 반환"""
    return list(_evidence_map.values())
