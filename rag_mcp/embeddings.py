from functools import lru_cache

import numpy as np

from config.settings import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    """Load the configured sentence-transformer once inside the RAG process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('Install sentence-transformers before starting the RAG MCP server.') from exc
    return SentenceTransformer(EMBEDDING_MODEL)


def encode_texts(texts: list[str]) -> np.ndarray:
    """Create normalized semantic embeddings suitable for cosine similarity."""
    if not texts:
        return np.empty((0, 0), dtype='float32')
    return get_embedder().encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype('float32')
