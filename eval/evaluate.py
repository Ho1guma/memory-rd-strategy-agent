"""Retrieval evaluation script — Hit Rate@K and MRR.

Usage:
    uv run python eval/evaluate.py --model BAAI/bge-m3 --k 5

Results are printed to stdout and optionally saved to eval/results.json.
Update README with the winning model's metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_golden(path: str = "eval/golden_queries.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    rr = [1 / r for r in ranks if r is not None]
    return sum(rr) / len(ranks) if ranks else 0.0


def evaluate_retrieval(model_name: str, k: int, use_hybrid: bool) -> dict:
    from rd_strategy_agent.agents.retrieve import hybrid_search, _get_collection, EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer

    golden = load_golden()
    queries = golden["queries"]
    eval_params = golden.get("eval_params", {})
    k_values = eval_params.get("k_values", [1, 3, 5, 10])

    results_per_k: dict[int, list[int | None]] = {kv: [] for kv in k_values}

    for q in queries:
        query_text = q["query"]
        relevant_keywords = [kw.lower() for kw in q["relevant_doc_keywords"]]

        if use_hybrid:
            chunks = hybrid_search(query_text, top_k=max(k_values))
            texts = [c["text"].lower() for c in chunks]
        else:
            collection = _get_collection()
            res = collection.query(query_texts=[query_text], n_results=max(k_values))
            texts = [t.lower() for t in res["documents"][0]]

        # Find rank of first relevant result
        rank = None
        for i, text in enumerate(texts, start=1):
            if any(kw in text for kw in relevant_keywords):
                rank = i
                break

        for kv in k_values:
            results_per_k[kv].append(rank)

    metrics = {
        "model": model_name,
        "hybrid": use_hybrid,
        "hit_rate": {f"@{kv}": hit_rate_at_k(results_per_k[kv], kv) for kv in k_values},
        "mrr": mrr(results_per_k[max(k_values)]),
        "n_queries": len(queries),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hybrid", action="store_true", default=True)
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    print(f"Evaluating: model={args.model}, k={args.k}, hybrid={args.hybrid}")
    metrics = evaluate_retrieval(args.model, args.k, args.hybrid)

    print(json.dumps(metrics, indent=2))

    out_path = Path(args.output)
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.append(metrics)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Results appended → {args.output}")


if __name__ == "__main__":
    main()
