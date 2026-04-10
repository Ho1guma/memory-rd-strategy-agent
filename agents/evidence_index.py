"""
Evidence Index (벡터 DB 래퍼)

WebSearch + AcademicSearch 수집 완료 후 evidence_store를 FAISS로 인덱싱.
Analysis/Report 에이전트가 company×technology별 쿼리로 관련 증거 top-K만 조회.

- 임베딩: BAAI/bge-m3 (retrieve agent와 동일 모델, 모듈 레벨 캐싱)
- 저장: 프로세스 내 전역 변수 (FAISS 객체는 직렬화 불가 → TypedDict state 저장 불가)
"""

import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agents.state import EvidenceItem

EMBEDDING_MODEL = os.environ.get("EVIDENCE_INDEX_MODEL", "all-MiniLM-L6-v2")

# 모듈 레벨 싱글턴: 프로세스 내에서 한 번만 로드
_embeddings: Optional[HuggingFaceEmbeddings] = None
_faiss_index: Optional[FAISS] = None
_evidence_map: dict[int, EvidenceItem] = {}  # FAISS doc id → EvidenceItem


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"[EvidenceIndex] 임베딩 모델 로드: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


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
        embeddings = _get_embeddings()
        _faiss_index = FAISS.from_documents(docs, embeddings)
        print(f"[EvidenceIndex] 인덱싱 완료 ({len(docs)}건)")
        return True
    except Exception as e:
        print(f"[EvidenceIndex] 인덱싱 실패: {e}")
        return False


def query_evidence(query: str, k: int = 8) -> list[EvidenceItem]:
    """
    쿼리와 관련된 증거 top-k를 반환.
    인덱스 미구축 시 빈 리스트 반환.
    """
    if _faiss_index is None or not _evidence_map:
        return []

    try:
        results = _faiss_index.similarity_search(query, k=min(k, len(_evidence_map)))
        evidence_items = []
        seen_idx = set()
        for doc in results:
            idx = doc.metadata.get("evidence_idx")
            if idx is not None and idx not in seen_idx and idx in _evidence_map:
                evidence_items.append(_evidence_map[idx])
                seen_idx.add(idx)
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
