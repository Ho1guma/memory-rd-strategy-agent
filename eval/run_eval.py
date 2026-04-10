"""
Retrieval 평가 스크립트
- Hit Rate@K: 상위 K개 결과 내 정답 키워드 포함 비율
- MRR: Mean Reciprocal Rank

사용법:
  uv run python eval/run_eval.py --k 5 --model BAAI/bge-m3
"""

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def hit_rate_at_k(results: list[str], relevant_keywords: list[str], k: int) -> float:
    """상위 K 결과 중 관련 키워드가 하나 이상 포함되면 hit"""
    top_k = results[:k]
    for text in top_k:
        if any(kw.lower() in text.lower() for kw in relevant_keywords):
            return 1.0
    return 0.0


def reciprocal_rank(results: list[str], relevant_keywords: list[str]) -> float:
    """첫 번째 관련 결과의 역순위"""
    for rank, text in enumerate(results, start=1):
        if any(kw.lower() in text.lower() for kw in relevant_keywords):
            return 1.0 / rank
    return 0.0


def run_evaluation(k: int = 5, model_name: str = "BAAI/bge-m3") -> dict:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi

    # 쿼리 로드
    with open("eval/golden_queries.json", encoding="utf-8") as f:
        queries = json.load(f)

    # 문서 로드
    data_path = Path("data")
    if not data_path.exists() or not any(data_path.iterdir()):
        print("⚠️  data/ 디렉토리가 비어 있습니다. 문서를 추가 후 재실행하세요.")
        return {}

    loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)

    if not chunks:
        print("⚠️  청크 없음. data/ 에 .txt 파일을 추가하세요.")
        return {}

    print(f"평가 시작 — 모델: {model_name}, K={k}, 청크: {len(chunks)}개, 쿼리: {len(queries)}개")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    results_dense = {"hit_rate": [], "mrr": []}
    results_bm25 = {"hit_rate": [], "mrr": []}
    results_hybrid = {"hit_rate": [], "mrr": []}

    for q in queries:
        query = q["query"]
        relevant = q["relevant_keywords"]

        # Dense
        dense_docs = vectorstore.similarity_search(query, k=k)
        dense_texts = [d.page_content for d in dense_docs]
        results_dense["hit_rate"].append(hit_rate_at_k(dense_texts, relevant, k))
        results_dense["mrr"].append(reciprocal_rank(dense_texts, relevant))

        # BM25
        scores = bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_texts = [chunks[i].page_content for i in top_idx]
        results_bm25["hit_rate"].append(hit_rate_at_k(bm25_texts, relevant, k))
        results_bm25["mrr"].append(reciprocal_rank(bm25_texts, relevant))

        # Hybrid (RRF)
        rrf: dict[str, float] = {}
        all_docs_map: dict[str, str] = {}
        for rank, doc in enumerate(dense_docs):
            key = doc.page_content[:80]
            rrf[key] = rrf.get(key, 0) + 1 / (rank + 60)
            all_docs_map[key] = doc.page_content
        for rank, idx in enumerate(top_idx):
            key = chunks[idx].page_content[:80]
            rrf[key] = rrf.get(key, 0) + 1 / (rank + 60)
            all_docs_map[key] = chunks[idx].page_content
        hybrid_ranked = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:k]
        hybrid_texts = [all_docs_map[key] for key in hybrid_ranked]
        results_hybrid["hit_rate"].append(hit_rate_at_k(hybrid_texts, relevant, k))
        results_hybrid["mrr"].append(reciprocal_rank(hybrid_texts, relevant))

    summary = {
        "model": model_name,
        "k": k,
        "n_queries": len(queries),
        "Dense": {
            "Hit Rate@K": round(sum(results_dense["hit_rate"]) / len(queries), 4),
            "MRR": round(sum(results_dense["mrr"]) / len(queries), 4),
        },
        "BM25": {
            "Hit Rate@K": round(sum(results_bm25["hit_rate"]) / len(queries), 4),
            "MRR": round(sum(results_bm25["mrr"]) / len(queries), 4),
        },
        "Hybrid (Dense+BM25, RRF)": {
            "Hit Rate@K": round(sum(results_hybrid["hit_rate"]) / len(queries), 4),
            "MRR": round(sum(results_hybrid["mrr"]) / len(queries), 4),
        },
    }

    print("\n── 평가 결과 ─────────────────────────────────────")
    for method, scores in summary.items():
        if isinstance(scores, dict):
            print(f"  {method}: Hit Rate@{k}={scores['Hit Rate@K']}, MRR={scores['MRR']}")
    print("──────────────────────────────────────────────────")
    print("→ 위 수치를 README.md Retrieval Evaluation 표에 반영하세요.\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()
    run_evaluation(k=args.k, model_name=args.model)
