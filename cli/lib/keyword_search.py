import math
from .preprocess import preprocess
from .constants import DEFAULT_SEARCH_LIMIT
from .inverted_index import InvertedIndex


def search_title(keyword: str, limit=DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    filtered_movies = []
    seen = set()
    keyword_tokens = preprocess(keyword)

    for keyword_token in keyword_tokens:
        doc_ids = idx.get_documents(keyword_token)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            movie = idx.docmap[doc_id]
            filtered_movies.append(movie)
            if len(filtered_movies) >= limit:
                return filtered_movies
    return filtered_movies


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
