# EvidenceEngine — model + terminology reference

Plain-English reference for every model, dataset, and technique in the project. For new contributors and future-you.

## The two big confusions to clear up first

**SciFact ≠ a model.** SciFact is a **dataset** — a collection of ~1,400 biomedical claims that humans manually labeled as "supported by evidence", "contradicted by evidence", or "not enough info". Released by AI2 (Allen Institute for AI) in 2020. Used to train and benchmark fact-checking systems on bio claims.

**NLI ≠ a model.** NLI is a **task type** — Natural Language Inference. Given a "premise" (some text) and a "hypothesis" (a claim), decide if the premise **entails** (supports), **contradicts**, or is **neutral** to the hypothesis. NLI is the core technique this project uses to verify claims.

So when the docs say "SciFact-tuned NLI model," it means a model that performs the NLI task, fine-tuned on the SciFact dataset to be good at biomedical claims specifically.

## Every model in the pipeline

### `cross-encoder/nli-deberta-v3-small` — the NLI model

- **What it is:** A neural network using the **DeBERTa-v3-small** architecture (released by Microsoft, ~140 million parameters). The "small" variant is the lightweight one. There's also `base` (~180M) and `large` (~435M).
- **What it does:** Given a (claim, evidence span) pair, outputs three probabilities — entailment, neutral, contradiction. Lives at `/data/models/scifact-nli` on the Fly.io volume. It's the LAST step of the pipeline that decides "supported / contradicted / etc."
- **"Cross-encoder":** Both inputs (claim + evidence) feed into the SAME network together. Reads them jointly and decides their relationship. The opposite is a "bi-encoder" which encodes each text separately into a vector — bi-encoders are faster but less accurate.
- **"Fine-tuned on SciFact":** Took the base DeBERTa weights, then trained further on SciFact biomedical claims so the model gets better at the *specific* style of biomedical assertions vs general English NLI.

### `BAAI/bge-base-en-v1.5` — the dense retriever / encoder

- **What it is:** A model from BAAI (Beijing Academy of AI), 110 million parameters. "BGE" = **B**AAI **G**eneral **E**mbedding. "base" = the medium-size variant (also `bge-small` and `bge-large`). "en" = English. "v1.5" = version.
- **What it does:** Takes any sentence (claim or evidence span) and produces a 768-dimensional vector — a list of 768 numbers that represents what the sentence "means". Two sentences with similar meanings have vectors that are mathematically close together (cosine similarity).
- **This is the "encoder"** — encodes text into vectors. Used during the **retrieval** stage: the system encodes the claim into a vector, encodes every paragraph in the cited PDF into vectors, and finds the paragraphs whose vectors are closest. That's "dense retrieval" (vectors are dense — every dimension has a value).
- **Important:** BGE-base requires the prefix `"Represent this sentence for searching relevant passages: "` on QUERY embeddings only. Document (corpus) embeddings must NOT get this prefix. Forgetting it causes a ~5 to 8 pp accuracy regression.

### `cross-encoder/ms-marco-MiniLM-L6-v2` — the reranker

- **What it is:** A small cross-encoder (22 million parameters) from Microsoft, trained on MS MARCO (a Microsoft search-relevance dataset). "MiniLM" is the architecture. "L6" = 6 layers (smaller = faster).
- **What it does:** After dense retrieval finds, say, 10 candidate passages, the reranker takes each `(claim, passage)` pair and outputs a relevance score. Sorts them. Takes the top 2-3 to send to NLI. Same cross-encoder pattern as the NLI model, but trained for "is this passage relevant?" not "does it entail this claim?". Two different jobs.

### `BM25` — keyword retrieval

- **Not a neural model at all.** BM25 is a 1990s statistics-based ranking function. Counts how many times each query word appears in each document, weighted by inverse document frequency. Fast, no training, no GPU.
- **What it does:** First-pass keyword retrieval. Cheap. Combined with the dense retriever (BGE) for hybrid search. "BM25-only" is the fallback when corpus exceeds the dense retrieval skip threshold (1500 chunks; see `core/config.py`).

## How the four work together in a single verification

A user uploads a claim like "Subcutaneous semaglutide reduced AF risk by 23% (Smith et al., 2023)".

1. **Claim extractor** (no model — just NLTK + regex) finds the sentence and detects "(Smith et al., 2023)" as a citation.
2. **Anchor resolver** (no model — fuzzy string matching) tries to match "Smith 2023" to one of the uploaded source PDFs.
3. **BM25** does keyword search of the matched source PDF for the claim text. Returns top 5 passages.
4. **BGE-base** (dense retriever) encodes the claim and all source PDF passages into 768-dim vectors. Finds top 5 closest passages by cosine similarity.
5. **MiniLM reranker** takes the union of BM25's 5 + BGE's 5 (≤10 passages), scores each `(claim, passage)` pair for relevance, picks top 2.
6. **DeBERTa NLI** (SciFact-tuned) runs `(passage, claim)` for each top passage. Outputs entail/neutral/contradiction probabilities. Highest wins. Threshold logic decides supported / contradicted / insufficient_support / needs_review.

Five distinct neural-or-statistical components. Each does one job.

## Other terms you'll see

| Term | Meaning |
|---|---|
| **Embedding** | A vector of numbers representing text meaning. BGE produces embeddings. |
| **Token** | Roughly a word fragment. Models process text as tokens, not characters. |
| **Fine-tune** | Take a pre-trained model and train it more on your specific task/data so it specializes. |
| **Inference** | Running the model to get a prediction (the opposite of training). |
| **Threshold** | A number that converts a probability into a decision. The NLI uses 0.92 for "supported" and 0.75 for "contradicted" (chosen by the threshold sweep in eval/run_nli_threshold_sweep.py). |
| **Entailment / Neutral / Contradiction** | The three NLI labels. Entailment = supports. Contradiction = contradicts. Neutral = neither. |
| **Premise / Hypothesis** | NLI terminology. In this project: premise = the evidence span (what the source paper says); hypothesis = the claim (what the user wrote). Order matters — flipping hurts accuracy. |
| **Pre-training vs fine-tuning** | Pre-training is the expensive process (months, GPUs, billions of words) where the model learns general language. Fine-tuning is cheap (hours, maybe one GPU, thousands of examples) where it specializes. |
| **Cross-encoder vs bi-encoder** | Cross-encoder reads two texts together (slow, accurate). Bi-encoder encodes each separately into a vector (fast, less accurate). The NLI and reranker are cross-encoders; BGE is a bi-encoder. |
| **Retrieval** | Finding the most relevant passages from a corpus given a query. The pipeline retrieves evidence spans for each claim. |
| **Dense vs sparse** | Dense retrieval uses neural embeddings (BGE). Sparse retrieval uses keyword counts (BM25). Hybrid combines both. |
| **Reranking** | A second-pass scoring step after retrieval. The reranker is more accurate than the retriever but slower, so it only scores the top candidates from retrieval. |
| **Calibration** | Whether a model's confidence numbers match its actual accuracy. Overconfident = says 90% but is right 75% of the time. |

## Active in production right now (Fly.io)

- BM25 (statistical keyword search) ✓
- BGE-base (dense embeddings) ✓
- MiniLM reranker ✓
- DeBERTa-v3-small NLI (SciFact-tuned) ✓ — weights at `/data/models/scifact-nli`

The four auto-load on Fly machine boot. No manual management required.
