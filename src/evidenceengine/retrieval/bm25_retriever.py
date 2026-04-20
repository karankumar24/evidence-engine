"""BM25 index build, query, and disk-cache operations using bm25s."""

import os


def build_index(spans: list[str], index_dir: str):
    """Build a BM25 index from span strings and save to disk.

    Args:
        spans: List of text strings to index (paragraph + sentence spans).
        index_dir: Directory path to save the index. Created if missing.

    Returns:
        Populated bm25s.BM25 retriever with the corpus loaded in memory.

    Raises:
        ValueError: If spans is empty — bm25s produces a broken index from
                    an empty corpus that silently returns no results on query.
    """
    if not spans:
        raise ValueError(
            "Cannot build BM25 index from empty corpus — "
            "caller must ensure at least one span is extracted before indexing."
        )
    import bm25s  # noqa: PLC0415 — lazy to avoid blocking startup
    os.makedirs(index_dir, exist_ok=True)
    corpus_tokens = bm25s.tokenize(spans, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(index_dir, corpus=spans)
    # retriever.corpus is not set by index/save — only by load(load_corpus=True).
    # Attach the spans directly so query_index can resolve hit IDs → passage text
    # without needing a subsequent load round-trip.
    retriever.corpus = [{"id": i, "text": s} for i, s in enumerate(spans)]
    return retriever


def load_or_build_index(spans: list[str], index_dir: str):
    """Load a BM25 index from disk if it exists; otherwise build and save it.

    Index is keyed by directory path, which callers should set to
    {settings.index_dir}/{source_document_id}/ for per-document caching.

    Args:
        spans: Span strings used to build the index if not already on disk.
        index_dir: Directory containing the saved index files.

    Returns:
        bm25s.BM25 retriever with corpus loaded (mmap=True for memory efficiency).
    """
    import bm25s  # noqa: PLC0415 — lazy to avoid blocking startup
    # bm25s.BM25.load requires the .index.npy file — check for it
    index_file = os.path.join(index_dir, "index.npy")
    if os.path.exists(index_file):
        return bm25s.BM25.load(index_dir, load_corpus=True, mmap=True)
    return build_index(spans, index_dir)


def query_index(retriever, query: str, k: int = 10) -> tuple[list[str], list[float]]:
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
    import bm25s  # noqa: PLC0415 — lazy to avoid blocking startup
    query_tokens = bm25s.tokenize([query], stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=k)
    # results shape: (1, k); scores shape: (1, k).
    # Each entry in results[0] is a numpy int index into retriever.corpus OR a
    # dict {id, text} when the corpus was loaded with load_corpus=True. We
    # resolve both to the actual text. Before this resolution the function
    # was returning the stringified indices (e.g. "4", "11") as if they were
    # passage text — retrieval was effectively broken for callers that use
    # the returned texts as evidence.
    corpus = getattr(retriever, "corpus", None)

    def _resolve(entry):
        if isinstance(entry, dict):
            return entry.get("text", "")
        if corpus is not None:
            try:
                item = corpus[int(entry)]
            except (TypeError, ValueError, IndexError):
                return str(entry)
            if isinstance(item, dict):
                return item.get("text", "")
            return str(item)
        return str(entry)

    span_texts = [_resolve(s) for s in results[0]]
    score_list = [float(s) for s in scores[0]]
    return span_texts, score_list
