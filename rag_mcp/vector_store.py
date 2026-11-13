from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from config.settings import EMBEDDING_MODEL, VECTOR_STORE_DIR, list_files, resolve_file
from rag_mcp.chunking import EXTENSIONS, chunk_source
from rag_mcp.embeddings import encode_texts


@dataclass
class SourceIndex:
    index: object
    chunks: list[dict]
    metadata: dict
    signature: tuple[int, int]


INDEXES: dict[tuple[str, str | None], SourceIndex] = {}


def import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('Install faiss-cpu before starting the RAG MCP server.') from exc
    return faiss


def source_signature(filename: str) -> tuple[int, int]:
    path = resolve_file(filename, EXTENSIONS)
    details = path.stat()
    return details.st_mtime_ns, details.st_size


def index_paths(filename: str, sheet: str | None) -> tuple:
    identity = json.dumps([filename, sheet], ensure_ascii=False).encode('utf-8')
    stem = sha256(identity).hexdigest()[:24]
    return VECTOR_STORE_DIR / f'{stem}.faiss', VECTOR_STORE_DIR / f'{stem}.json'


def save_index(filename: str, sheet: str | None, stored: SourceIndex) -> None:
    """Persist vectors and traceable chunks without exposing source names in paths."""
    index_path, metadata_path = index_paths(filename, sheet)
    import_faiss().write_index(stored.index, str(index_path))
    metadata_path.write_text(json.dumps({
        'filename': filename,
        'sheet': sheet,
        'signature': list(stored.signature),
        'embedding_model': EMBEDDING_MODEL,
        'chunks': stored.chunks,
        'metadata': stored.metadata,
    }, ensure_ascii=False), encoding='utf-8')


def load_saved_index(filename: str, sheet: str | None) -> SourceIndex | None:
    """Reuse a valid persisted index; changed sources are rebuilt automatically."""
    index_path, metadata_path = index_paths(filename, sheet)
    if not index_path.is_file() or not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
        signature = tuple(payload['signature'])
        if signature != source_signature(filename) or payload['embedding_model'] != EMBEDDING_MODEL:
            return None
        return SourceIndex(import_faiss().read_index(str(index_path)), payload['chunks'],
                           payload['metadata'], signature)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def load_source(filename: str, sheet: str | None = None) -> dict:
    """Build or refresh the semantic FAISS index for one source."""
    chunks, metadata = chunk_source(filename, sheet)
    if not chunks:
        raise ValueError('The source contains no searchable content.')
    embeddings = encode_texts([chunk['text'] for chunk in chunks])
    if embeddings.ndim != 2 or not embeddings.shape[1]:
        raise ValueError('The embedding model returned no vectors.')
    faiss = import_faiss()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype='float32'))
    key = (filename, sheet)
    INDEXES[key] = SourceIndex(index, chunks, metadata, source_signature(filename))
    save_index(filename, sheet, INDEXES[key])
    return {**metadata, 'indexed': len(chunks), 'embedding_dimension': embeddings.shape[1],
            'similarity': 'cosine similarity over normalized sentence embeddings'}


def get_index(filename: str, sheet: str | None = None) -> SourceIndex:
    key = (filename, sheet)
    cached = INDEXES.get(key)
    if cached is None:
        cached = load_saved_index(filename, sheet)
        if cached is not None:
            INDEXES[key] = cached
    if cached is None or cached.signature != source_signature(filename):
        load_source(filename, sheet)
    return INDEXES[key]


def search_source(query: str, filename: str, top_k: int = 5,
                  sheet: str | None = None, min_score: float = 0.15) -> dict:
    """Retrieve semantically related rows/passages with source metadata."""
    if not query.strip() or not 1 <= top_k <= 20:
        raise ValueError('Provide a query and top_k between 1 and 20.')
    if not -1 <= min_score <= 1:
        raise ValueError('min_score must be between -1 and 1.')
    stored = get_index(filename, sheet)
    query_vector = encode_texts([query])
    scores, positions = stored.index.search(query_vector.astype('float32'),
                                            min(top_k, len(stored.chunks)))
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0 and score >= min_score:
            results.append({**stored.chunks[position], 'score': float(score)})
    return {'filename': filename, 'results': results,
            'retrieval': 'semantic sentence embeddings with normalized cosine similarity'}


def source_info(filename: str, sheet: str | None = None) -> dict:
    """Return source metadata without downloading or running an embedding model."""
    _, metadata = chunk_source(filename, sheet)
    key = (filename, sheet)
    cached = INDEXES.get(key) or load_saved_index(filename, sheet)
    if cached is not None:
        INDEXES[key] = cached
    return {**metadata, 'indexed': cached is not None}


def search_documents(query: str, filename: str | None = None, top_k: int = 5) -> dict:
    """Backward-compatible search across one source or all supported sources."""
    names = [filename] if filename else list_files(EXTENSIONS)
    if not names:
        return {'results': [], 'message': 'No supported sources are available.'}
    results = []
    for name in names:
        results.extend(search_source(query, name, top_k)['results'])
    results.sort(key=lambda item: item['score'], reverse=True)
    return {'results': results[:top_k],
            'retrieval': 'semantic sentence embeddings with normalized cosine similarity'}
