"""
Retrieve Agent (T2, T3)
- data/ 디렉토리의 로컬 문서를 청킹 후 FAISS 인덱싱
- Hybrid Search: Dense (BGE-m3) + BM25, RRF로 결합
- 결과를 evidence_store 포맷으로 반환 → Supervisor에 보고
"""

import os
import json
from datetime import date
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from agents.state import AgentState, EvidenceItem


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
TOP_K = int(os.environ.get("TOP_K", 5))
DATA_DIR = "data"


def _build_queries(scope: dict, iteration_count: int) -> list[str]:
    """재시도 시 쿼리를 점진적으로 확장"""
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    keywords = scope.get("keywords", [])

    base_queries = [f"{tech} technology" for tech in technologies]
    competitor_queries = [f"{comp} {tech}" for comp in competitors for tech in technologies]

    if iteration_count == 0:
        return base_queries + keywords[:3]
    elif iteration_count == 1:
        return base_queries + competitor_queries[:5] + keywords
    else:
        # 마지막 재시도: 더 넓은 쿼리
        return base_queries + competitor_queries + keywords + [f"{tech} roadmap" for tech in technologies]


def _tag_item(snippet: str, title: str, scope: dict) -> tuple[list[str], list[str]]:
    """간단한 규칙 기반 키워드·엔티티 태깅 (SC1-2 충족용)"""
    technologies = scope.get("technologies", [])
    competitors = scope.get("competitors", [])
    text = (snippet + " " + title).lower()

    kw = [tech for tech in technologies if tech.lower() in text]
    entities = [comp for comp in competitors if comp.lower() in text]
    return kw, entities


def retrieve_agent(state: AgentState) -> dict:
    scope = state.get("scope", {})
    iteration_count = state.get("iteration_count", 0)
    queries = _build_queries(scope, iteration_count)

    data_path = Path(DATA_DIR)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"[Retrieve] data/ 디렉토리가 비어 있음 — 로컬 검색 스킵")
        return {}

    print(f"[Retrieve] 임베딩 모델 로드: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 문서 로드 및 청킹
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
    docs = loader.load()
    if not docs:
        print("[Retrieve] 로드된 문서 없음 — 스킵")
        return {}

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)
    print(f"[Retrieve] 청크 수: {len(chunks)}")

    # Dense retrieval (FAISS)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # BM25 retrieval
    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    new_evidence: list[EvidenceItem] = []
    seen_snippets = {item["snippet"][:100] for item in state.get("evidence_store", [])}

    for query in queries:
        # Dense 결과
        dense_results = vectorstore.similarity_search(query, k=TOP_K)

        # BM25 결과
        bm25_scores = bm25.get_scores(query.lower().split())
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
        bm25_results = [chunks[i] for i in top_bm25_idx]

        # RRF 결합 (간단 구현)
        rrf_scores: dict[str, float] = {}
        for rank, doc in enumerate(dense_results):
            key = doc.page_content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)

        all_docs = {d.page_content[:100]: d for d in dense_results + bm25_results}
        ranked = sorted(all_docs.keys(), key=lambda k: rrf_scores.get(k, 0), reverse=True)

        for key in ranked[:TOP_K]:
            doc = all_docs[key]
            snippet = doc.page_content[:500]
            if snippet[:100] in seen_snippets:
                continue
            seen_snippets.add(snippet[:100])
            kw, entities = _tag_item(snippet, doc.metadata.get("source", ""), scope)
            new_evidence.append(EvidenceItem(
                url=doc.metadata.get("source", "local"),
                title=doc.metadata.get("source", "local document"),
                date=doc.metadata.get("date", date.today().isoformat()),
                snippet=snippet,
                domain="local",
                keywords=kw,
                entities=entities,
            ))

    print(f"[Retrieve] 신규 증거 {len(new_evidence)}건 추가")
    return {"evidence_store": new_evidence}
