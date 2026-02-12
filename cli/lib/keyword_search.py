import math
from .constants import DEFAULT_SEARCH_LIMIT
from .inverted_index import InvertedIndex


def search_title(keyword: str, limit=DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    return idx.search(keyword, limit)


def bm25_search_title(term: str, limit=DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(term, limit)


def build():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def get_tf(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)


def get_bm25_tf(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term)


def get_idf(term: str):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_idf(term)


def get_tfidf(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tfidf(doc_id, term)


def get_bm25_idf(term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)
