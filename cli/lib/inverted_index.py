from collections import Counter
from .preprocess import preprocess
from pathlib import Path
from .search_utils import import_json
from .constants import TITLE_KEY, DESCRIPTION_KEY, ID_KEY, CACHE_PATH, BM25_K1, BM25_B
import pickle
import os
import math


class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")
        self.doc_lengths = dict()

    def __add_document(self, doc_id, text) -> None:
        tokens = preprocess(text)
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        res = 0.0
        for length in self.doc_lengths.values():
            res += length
        if self.doc_lengths:
            return res / len(self.doc_lengths.keys())
        else:
            return res

    def get_documents(self, term) -> list[str]:
        res = []
        term = term.lower()
        if term in self.index:
            for doc_id in self.index[term]:
                res.append(doc_id)

        res.sort()
        return res

    def get_tf(self, doc_id, term) -> int:
        term = preprocess(term)
        if len(term) > 1:
            raise Exception("More than one number token provided")
        term = term[0]
        doc_id_counter = self.term_frequencies.get(doc_id)
        res = 0
        if doc_id_counter:
            res = doc_id_counter.get(term)
        return res if res else 0

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        res = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return res

    def get_idf(self, term: str) -> float:
        tokens = preprocess(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = preprocess(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_tfidf(self, doc_id, term) -> float:
        term = preprocess(term)
        if len(term) > 1:
            raise Exception("More than one number token provided")
        term = term[0]
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def search(self, term, limit):
        filtered_movies = []
        seen = set()
        keyword_tokens = preprocess(term)

        for keyword_token in keyword_tokens:
            doc_ids = self.get_documents(keyword_token)
            for doc_id in doc_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                movie = self.docmap[doc_id]
                filtered_movies.append(movie)
                if len(filtered_movies) >= limit:
                    return filtered_movies
        return filtered_movies

    def bm25_search(self, query, limit) -> list[tuple]:
        query_tokens = preprocess(query)
        scores = dict()
        for token in query_tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                if not scores.get(doc_id):
                    scores[doc_id] = 0
                scores[doc_id] += self.bm25(doc_id, token)

        sorted_results = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)
        filtered_movies = []
        for result in sorted_results:
            filtered_movies.append((self.docmap[result[0]], result[1]))
            if len(filtered_movies) == limit:
                break
        return filtered_movies

    def build(self) -> None:
        movies = import_json()
        for movie in movies:
            title, description, id = (
                movie[TITLE_KEY],
                movie[DESCRIPTION_KEY],
                movie[ID_KEY],
            )
            self.docmap[id] = movie
            text = f"{title} {description}"
            self.__add_document(id, text)

    def save(self) -> None:
        cache_path = Path(CACHE_PATH)
        if not cache_path.is_dir():
            os.mkdir(CACHE_PATH)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
            file.close()

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)
            file.close()

        with open(self.tf_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
            file.close()

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)
            file.close()

    def load(self):
        if not Path.is_file(Path(self.index_path)):
            raise Exception("File not found")
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        if not Path.is_file(Path(self.docmap_path)):
            raise Exception("File not found")
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        if not Path.is_file(Path(self.tf_path)):
            raise Exception("File not found")
        with open(self.tf_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        if not Path.is_file(Path(self.doc_lengths_path)):
            raise Exception("File not found")
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)
