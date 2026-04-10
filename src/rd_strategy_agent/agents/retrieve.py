"""Retrieve Agent — Tasks T2, T3.

Indexes evidence_store into a vector+BM25 hybrid retrieval system.
Outcome: semantic search over collected evidence is available.
"""
from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rd_strategy_agent.state import AgentState, EvidenceItem

# Model selected via eval/evaluate.py (Hit Rate@K / MRR benchmark)
# Update README when a better model is confirmed.
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "rd_evidence"
PERSIST_DIR = "chroma_db"


def _get_collection() -> chromadb.Collection:
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)


def retrieve_index(state: AgentState) -> dict:
    """T2/T3: Index evidence_store into ChromaDB (dense) + build BM25 index."""
    evidence: list[EvidenceItem] = state["evidence_store"]
    if not evidence:
        return {}

    collection = _get_collection()
    docs, ids, metadatas = [], [], []
    for i, ev in enumerate(evidence):
        doc_id = f"ev_{i}_{hash(ev['url']) & 0xFFFF:04x}"
        docs.append(ev["snippet"])
        ids.append(doc_id)
        metadatas.append({
            "url": ev["url"],
            "title": ev["title"],
            "date": ev["date"],
            "domain": ev["domain"],
            "keywords": json.dumps(ev["keywords"]),
            "entities": json.dumps(ev["entities"]),
        })

    # Upsert avoids duplicates on retry
    collection.upsert(documents=docs, ids=ids, metadatas=metadatas)

    # BM25 index (in-memory, rebuilt each run)
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    # Persist BM25 corpus alongside ChromaDB for evaluation
    bm25_path = Path(PERSIST_DIR) / "bm25_corpus.json"
    bm25_path.parent.mkdir(exist_ok=True)
    bm25_path.write_text(json.dumps({"docs": docs, "ids": ids}))

    print(f"[Retrieve] Indexed {len(docs)} documents into ChromaDB + BM25.")
    return {}


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """Hybrid dense+BM25 retrieval with Reciprocal Rank Fusion (RRF)."""
    collection = _get_collection()

    # Dense retrieval
    dense_results = collection.query(query_texts=[query], n_results=top_k)
    dense_ids = dense_results["ids"][0]
    dense_metas = dense_results["metadatas"][0]
    dense_docs = dense_results["documents"][0]

    # BM25 retrieval
    bm25_path = Path(PERSIST_DIR) / "bm25_corpus.json"
    if not bm25_path.exists():
        return [{"id": i, "text": d, "meta": m} for i, d, m in zip(dense_ids, dense_docs, dense_metas)]

    corpus = json.loads(bm25_path.read_text())
    tokenized = [d.lower().split() for d in corpus["docs"]]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(
        zip(corpus["ids"], corpus["docs"], scores),
        key=lambda x: x[2], reverse=True
    )[:top_k]

    # RRF fusion
    k = 60
    rrf: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _, _) in enumerate(bm25_ranked):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    # Build result list
    id_to_meta = dict(zip(dense_ids, dense_metas))
    id_to_doc = dict(zip(dense_ids, dense_docs))
    bm25_meta = {}  # meta not stored in bm25; dense coverage sufficient for now
    results = []
    for doc_id in sorted_ids:
        results.append({
            "id": doc_id,
            "text": id_to_doc.get(doc_id, ""),
            "meta": id_to_meta.get(doc_id, {}),
            "rrf_score": rrf[doc_id],
        })
    return results
