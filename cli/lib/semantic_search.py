import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

from .constants import (
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    DESCRIPTION_KEY,
    SCORE_KEY,
    TITLE_KEY,
)
from .search_utils import import_json


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
        self.embedding_path = os.path.join(CACHE_PATH, "embeddings.npy")

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        similarity_scores = []
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        q_embedding = self.generate_embedding(query)
        for index, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(embedding, q_embedding)
            similarity_scores.append((similarity, self.documents[index]))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        limit = min(limit, len(similarity_scores))
        res = []
        for index in range(limit):
            curr_res = {}
            curr_res[SCORE_KEY] = similarity_scores[index][0]
            curr_res[TITLE_KEY] = similarity_scores[index][1][TITLE_KEY]
            curr_res[DESCRIPTION_KEY] = similarity_scores[index][1][DESCRIPTION_KEY]
            res.append(curr_res)
        return res

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Empty text cannot have embeddings")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        string_rep = []

        for document in self.documents:
            if not document.get("id"):
                continue
            self.document_map[document["id"]] = document

            if not document.get("title") or not document.get("description"):
                continue
            string_rep.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(string_rep, show_progress_bar=True)
        self.save()
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        embed_path = Path(self.embedding_path)
        print("Does the embed path exist", embed_path.is_file())
        if not embed_path.is_file():
            return self.build_embeddings(documents)

        self.embeddings = np.load(file=self.embedding_path)
        if len(self.embeddings) == len(documents):
            self.documents = documents
            for document in documents:
                if not document.get("id"):
                    continue
                self.document_map[document["id"]] = document
            return self.embeddings
        else:
            return self.build_embeddings(documents)

    def save(self) -> None:
        cache_path = Path(CACHE_PATH)
        if not cache_path.is_dir():
            os.mkdir(CACHE_PATH)

        if self.embeddings is None:
            raise ValueError("No embeddings to store")
        with open(self.embedding_path, "wb") as file:
            np.save(arr=self.embeddings, file=file)


def verify_model():
    sem_search = SemanticSearch()

    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def embed_text(text):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    sem_search = SemanticSearch()
    documents = import_json()
    embeddings = sem_search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2) -> float:
    # print(np.linalg.norm(vec1))
    # print(np.isnan(vec1).any(), np.isinf(vec1).any())
    # print(np.linalg.norm(vec2))
    # print(np.isnan(vec2).any(), np.isinf(vec2).any())
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search(query: str, limit=DEFAULT_SEARCH_LIMIT):
    sem_search = SemanticSearch()
    documents = import_json()
    sem_search.load_or_create_embeddings(documents)
    results = sem_search.search(query, limit)
    for index, movie in enumerate(results):
        print(f"{index}. {movie[TITLE_KEY]} (score: {movie[SCORE_KEY]:.4f})")
        # print(movie[DESCRIPTION_KEY])
        print("\n")
