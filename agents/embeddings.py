"""
공유 임베딩 모델 싱글턴

retrieve.py와 evidence_index.py가 동일 HuggingFaceEmbeddings 인스턴스를 재사용.
bge-m3은 570MB 모델이므로 프로세스 내에서 한 번만 로드.
"""

import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"[Embeddings] 모델 로드: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings
