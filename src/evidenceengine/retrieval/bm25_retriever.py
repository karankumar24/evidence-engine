"""BM25 index build, query, and disk-cache operations using bm25s."""

import os

import bm25s


def build_index(spans: list[str], index_dir: str) -> bm25s.BM25:
    """Build a BM25 index from span strings and save to disk.

    Args:
        spans: List of text strings to index (paragraph + sentence spans).
        index_dir: Directory path to save the index. Created if missing.

    Returns:
        Populated bm25s.BM25 retriever with the corpus loaded in memory.
    """
    os.makedirs(index_dir, exist_ok=True)
    corpus_tokens = bm25s.tokenize(spans, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(index_dir, corpus=spans)
    return retriever


def load_or_build_index(spans: list[str], index_dir: str) -> bm25s.BM25:
    """Load a BM25 index from disk if it exists; otherwise build and save it.

    Index is keyed by directory path, which callers should set to
    {settings.index_dir}/{source_document_id}/ for per-document caching.

    Args:
        spans: Span strings used to build the index if not already on disk.
        index_dir: Directory containing the saved index files.

    Returns:
        bm25s.BM25 retriever with corpus loaded (mmap=True for memory efficiency).
    """
    # bm25s.BM25.load requires the .index.npy file — check for it
    index_file = os.path.join(index_dir, "index.npy")
    if os.path.exists(index_file):
        return bm25s.BM25.load(index_dir, load_corpus=True, mmap=True)
    return build_index(spans, index_dir)


def query_index(retriever: bm25s.BM25, query: str, k: int = 10) -> tuple[list[str], list[float]]:
    """Query a BM25 index for the top-k most relevant spans.

    bm25s.retrieve() returns 2D numpy arrays of shape (n_queries, k).
    We always issue a single query so we slice [0] to get flat lists.

    Args:
        retriever: Populated bm25s.BM25 retriever.
        query: Query string (claim text).
        k: Number of top results to return.

    Returns:
        Tuple of (span_texts, scores) as plain Python lists.
        len(span_texts) == len(scores) == min(k, corpus_size).
    """
    query_tokens = bm25s.tokenize([query], stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=k)
    # results shape: (1, k); scores shape: (1, k) — slice [0] for single-query
    span_texts = [str(s) for s in results[0]]
    score_list = [float(s) for s in scores[0]]
    return span_texts, score_list
