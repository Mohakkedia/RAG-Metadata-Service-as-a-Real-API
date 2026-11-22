"""
embeddings.py
---------------
Produces fixed-dimension embeddings for text chunks using scikit-learn's
`HashingVectorizer` — a deliberately *stateless* feature extraction method
(no vocabulary to fit, no corpus-wide refit needed) chosen specifically so
`/ingest` can genuinely be an incremental, one-document-at-a-time API
endpoint, which a TF-IDF vectorizer (fit-once-on-a-corpus) cannot support
without periodic full reindexing.

This sandbox has no route to huggingface.co, so a downloaded dense
embedding model (sentence-transformers) isn't available; swapping one in
is a ~10-line change confined to this file — everything else in the
service calls `embed(text) -> np.ndarray` and doesn't know or care how.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_DIM = 1024

_vectorizer = HashingVectorizer(
    n_features=EMBEDDING_DIM, alternate_sign=False, norm="l2",
    ngram_range=(1, 2), stop_words="english",
)


def embed(text: str) -> np.ndarray:
    vec = _vectorizer.transform([text])
    return vec.toarray()[0].astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    vecs = _vectorizer.transform(texts)
    return vecs.toarray().astype(np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """doc_vecs: (n, dim). query_vec, doc_vecs are already L2-normalized by
    the vectorizer (norm='l2'), so cosine similarity reduces to a dot product."""
    if doc_vecs.shape[0] == 0:
        return np.array([])
    return doc_vecs @ query_vec
