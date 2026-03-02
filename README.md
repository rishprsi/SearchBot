# SearchBot

A multi-strategy movie search engine that implements a full information retrieval pipeline — from keyword search with BM25, through semantic search with sentence-transformers, to hybrid search with Reciprocal Rank Fusion. It layers on LLM-powered features (query enhancement, reranking, RAG) via Google Gemini, adds multimodal image search with CLIP, and includes an evaluation framework with precision, recall, and F1 metrics.

All interaction is via CLI tools that search over a movie database for a fictional "Hoopla" streaming service.

## Features

- **Keyword Search** — Inverted index with TF, IDF, TF-IDF, and BM25 scoring
- **Semantic Search** — Dense vector embeddings via `all-MiniLM-L6-v2` with cosine similarity
- **Chunked Semantic Search** — Sentence-boundary chunking for finer-grained matching
- **Hybrid Search** — Weighted fusion and Reciprocal Rank Fusion (RRF) of keyword + semantic results
- **LLM Query Enhancement** — Spell correction, query rewriting, and synonym expansion via Gemini
- **Reranking** — Individual LLM scoring, batch LLM reranking, and cross-encoder neural reranking
- **RAG (Retrieval-Augmented Generation)** — Natural language answers, summaries, and cited responses powered by Gemini
- **Multimodal Search** — Search movies by image using CLIP (`clip-ViT-B-32`)
- **Search Evaluation** — Precision@k, Recall@k, and F1 scoring against a golden dataset

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Runtime |
| [uv](https://github.com/astral-sh/uv) | Package manager |
| [sentence-transformers](https://www.sbert.net/) | Semantic embeddings & cross-encoder reranking |
| [Google Gemini](https://ai.google.dev/) | LLM for RAG, query enhancement, reranking, image description |
| [NLTK](https://www.nltk.org/) | Porter Stemmer for text preprocessing |
| NumPy | Vector operations and embedding storage |
| Pillow | Image loading for multimodal search |
| CLIP (`clip-ViT-B-32`) | Multimodal image + text embeddings |

## Project Structure

```
SearchBot/
├── cli/                              # CLI entry point scripts
│   ├── keyword_search_cli.py         # Keyword/BM25 search
│   ├── semantic_search_cli.py        # Semantic embedding search
│   ├── hybrid_search_cli.py          # Hybrid (keyword + semantic) search
│   ├── augmented_generation_cli.py   # RAG generation
│   ├── multimodal_search_cli.py      # Image-based search
│   ├── describe_image_cli.py         # LLM image description / query rewriting
│   ├── evaluation_cli.py             # Search quality evaluation
│   ├── test_gemini.py                # Gemini API connectivity test
│   └── lib/                          # Core library modules
│       ├── constants.py              # Paths, scoring keys, config constants
│       ├── preprocess.py             # Text preprocessing pipeline
│       ├── search_utils.py           # Data loading, text chunking utilities
│       ├── inverted_index.py         # InvertedIndex (TF, IDF, TF-IDF, BM25)
│       ├── keyword_search.py         # Keyword search facade
│       ├── semantic_search.py        # SemanticSearch class
│       ├── chunked_semantic_search.py# ChunkedSemanticSearch class
│       ├── hybrid_search.py          # HybridSearch (fusion, reranking, enhancement)
│       ├── augmented_generation.py   # Core RAG implementation
│       ├── augmented_generation_cli.py # Extended RAG (summarize, citations, Q&A)
│       ├── multimodal_search.py      # CLIP-based image search
│       └── argparse_util.py          # Argparse builder utility
├── data/
│   ├── movies.json                   # Movie corpus (~25k lines)
│   ├── golden_dataset.json           # Ground-truth evaluation test cases
│   ├── stopwords.txt                 # 198 English stopwords
│   └── *.jpeg / *.png               # Sample images for multimodal testing
├── cache/                            # Generated at runtime (gitignored)
│   ├── *.pkl                         # Pickled inverted index, doc map, term freqs
│   ├── *.npy                         # NumPy embedding arrays
│   └── chunk_metadata.json           # Chunk-to-document mapping
├── pyproject.toml
└── uv.lock
```

## Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- A [Google Gemini API key](https://ai.google.dev/)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd SearchBot
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-3-flash-preview
   ```

   The Gemini API key is required for RAG, query enhancement, LLM reranking, and image description features. Keyword and semantic search work without it.

## Usage

All CLI scripts are run from the project root using `uv run`.

### Keyword Search

```bash
# Build the inverted index
uv run cli/keyword_search_cli.py build

# Search using BM25
uv run cli/keyword_search_cli.py bm25search "adventure movies with dinosaurs"

# Basic keyword search
uv run cli/keyword_search_cli.py search "space exploration"

# Inspect scoring components
uv run cli/keyword_search_cli.py tf <doc_id> <term>
uv run cli/keyword_search_cli.py idf <term>
uv run cli/keyword_search_cli.py tfidf <doc_id> <term>
```

### Semantic Search

```bash
# Semantic search (embeddings built on first run, cached for reuse)
uv run cli/semantic_search_cli.py search "heartwarming family film"

# Chunked semantic search
uv run cli/semantic_search_cli.py search_chunked "coming of age story"

# Inspect embeddings and chunking
uv run cli/semantic_search_cli.py embed_text "some text"
uv run cli/semantic_search_cli.py chunk "long text to chunk"
uv run cli/semantic_search_cli.py semantic_chunk "text to semantically chunk"
```

### Hybrid Search

```bash
# Weighted fusion search
uv run cli/hybrid_search_cli.py weighted-search "wizard magic school" --alpha 0.5

# RRF search with query enhancement and reranking
uv run cli/hybrid_search_cli.py rrf-search "bear family london" \
  --enhance spell \
  --rerank-method cross_encoder \
  --evaluate
```

**Enhancement modes:** `spell`, `rewrite`, `expand`
**Rerank methods:** `individual`, `batch`, `cross_encoder`

### RAG (Retrieval-Augmented Generation)

```bash
# Basic RAG answer
uv run cli/augmented_generation_cli.py rag "What are some good sci-fi movies?"

# Summarize search results
uv run cli/augmented_generation_cli.py summarize "dinosaur movies"

# Answer with citations
uv run cli/augmented_generation_cli.py citations "movies about bears"

# Conversational Q&A
uv run cli/augmented_generation_cli.py question "What should I watch tonight?"
```

### Multimodal Search

```bash
# Search movies by image
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg

# Verify image embedding
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
```

### Image-Based Query Rewriting

```bash
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "movies like this"
```

### Search Evaluation

```bash
# Evaluate search quality against the golden dataset
uv run cli/evaluation_cli.py --limit 5
```

Computes Precision@k, Recall@k, and F1 scores for predefined test queries.

## How It Works

### Search Pipeline

1. **Preprocessing** — Text is lowercased, punctuation is removed, stopwords are filtered, and tokens are stemmed using the Porter Stemmer.

2. **Indexing** — An inverted index maps stemmed tokens to document IDs with term frequency counts. Document embeddings are computed with `all-MiniLM-L6-v2` and cached as `.npy` files.

3. **Retrieval** — Keyword search uses BM25 (k1=1.5, b=0.75). Semantic search computes cosine similarity between query and document embeddings.

4. **Fusion** — Hybrid search combines both result lists using either weighted score normalization or Reciprocal Rank Fusion (`1/(k + rank)`, k=60).

5. **Enhancement** — Queries can be pre-processed by Gemini to fix typos, improve specificity, or add synonyms.

6. **Reranking** — Results can be reranked using LLM-based scoring (individual or batch) or a cross-encoder model (`cross-encoder/ms-marco-TinyBERT-L2-v2`).

7. **Generation** — For RAG, top search results are passed as context to Gemini, which generates natural language answers with optional citations.

### Caching

Inverted indexes and embeddings are computed once and cached in the `cache/` directory. Subsequent runs load from cache for fast startup.

## Data

The movie corpus (`data/movies.json`) contains documents with `id`, `title`, and `description` fields. The golden dataset (`data/golden_dataset.json`) provides ground-truth relevance judgments for evaluating search quality across various query categories (bears, wizards, dinosaurs, etc.).
