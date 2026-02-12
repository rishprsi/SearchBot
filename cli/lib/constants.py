import os

# Keyword search
DEFAULT_SEARCH_LIMIT = 5
TITLE_KEY = "title"
DESCRIPTION_KEY = "description"
ID_KEY = "id"
MOVIEKEY = "movies"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
GOLDEN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

SCORE_KEY = "score"
TITLE_KEY = "title"
DESCRIPTION_KEY = "description"

BM25_K1 = 1.5
BM25_B = 0.75

# Chunked Semantic search
MOVIE_INDEX = "movie_idx"
CHUNK_INDEX = "chunk_idx"
TOTAL_CHUNKS = "total_chunks"
CHUNK_KEY = "chunk"
TOTAL_CHUNKS = "total_chunks"
DESCRIPTION_LEN = "description_len"
TOTAL_DOCUMENTS = "total_documents"
SCORE_PRECISION = 4
DOCUMENT_KEY = "document"

BM25_SCORE = "bm25_score"
SEM_SCORE = "semantic_score"
HYBRID_SCORE = "hybrid_score"

BM25_RANK = "bm25_rank"
SEM_RANK = "sem_rank"
HYBRID_RANK = "hybrid_rank"
LLM_SCORE = "llm_score"
RRF_SCORE = "rrf_score"

TESTCASE_KEY = "test_cases"
QUERY_KEY = "query"
RELEVANT_DOCS = "relevant_docs"
