import os

DEFAULT_SEARCH_LIMIT = 5
TITLE_KEY = "title"
DESCRIPTION_KEY = "description"
ID_KEY = "id"
MOVIEKEY = "movies"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

SCORE_KEY = "score"
TITLE_KEY = "title"
DESCRIPTION_KEY = "description"

BM25_K1 = 1.5
BM25_B = 0.75
