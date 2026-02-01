from collections import Counter
from .preprocess import preprocess
from pathlib import Path
from .search_utils import import_json
from .constants import TITLE_KEY, DESCRIPTION_KEY, ID_KEY, CACHE_PATH, BM25_K1
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

    def __add_document(self, doc_id, text) -> None:
        tokens = preprocess(text)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)

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
        return res

    def get_bm25_tf(self, doc_id, term) -> float:
        tf = self.get_tf(doc_id, term)
        k1 = BM25_K1
        res = (tf * (k1 + 1)) / (tf + k1)
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
