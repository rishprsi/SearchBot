import os
from pathlib import Path
import numpy as np
import json

from lib.constants import (
    CACHE_PATH,
    CHUNK_INDEX,
    CHUNK_KEY,
    DESCRIPTION_KEY,
    DESCRIPTION_LEN,
    DOCUMENT_KEY,
    MOVIE_INDEX,
    SCORE_PRECISION,
    TITLE_KEY,
    TOTAL_CHUNKS,
)
from lib.search_utils import import_json, semantic_chunk_text
from .semantic_search import SemanticSearch, cosine_similarity


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_PATH, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        print("Building new chunk embeddings")
        self.documents = documents
        chunks: list[str] = []
        metadata: list[dict] = []
        description_len = 0
        for index, document in enumerate(documents):
            if document.get("id") is None or not document.get(DESCRIPTION_KEY):
                continue
            description_len += len(document[DESCRIPTION_KEY])
            self.document_map[document["id"]] = document

            curr_chunks: list[str] = semantic_chunk_text(
                document[DESCRIPTION_KEY], 4, 1
            )
            chunks.extend(curr_chunks)
            total_chunks = len(curr_chunks)
            for chunk_index in range(len(curr_chunks)):
                chunk_dict = {
                    MOVIE_INDEX: index,
                    CHUNK_INDEX: chunk_index,
                    TOTAL_CHUNKS: total_chunks,
                }
                metadata.append(chunk_dict)

        print("Total chunks are: ", len(chunks))
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        cache_path = Path(CACHE_PATH)
        if not cache_path.is_dir():
            os.mkdir(CACHE_PATH)

        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings to save")
        with open(self.chunk_embeddings_path, "wb") as file:
            np.save(file, self.chunk_embeddings)

        if self.chunk_metadata is None:
            raise ValueError("No chunk metadata to save")
        with open(self.chunk_metadata_path, "w") as file:
            json.dump(
                {
                    CHUNK_KEY: self.chunk_metadata,
                    TOTAL_CHUNKS: len(chunks),
                    DESCRIPTION_LEN: description_len,
                },
                file,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        print("Loading or creating embeddings")
        if (
            Path(self.chunk_metadata_path).is_file()
            and Path(self.chunk_embeddings_path).is_file()
        ):
            description_len = 0
            self.documents = documents
            for document in documents:
                if document.get("id") is None or not document.get(DESCRIPTION_KEY):
                    continue
                self.document_map[document["id"]] = document
                description_len += len(document[DESCRIPTION_KEY])
            with open(self.chunk_metadata_path, "r") as file:
                metadata = json.load(file)

            self.chunk_metadata = metadata[CHUNK_KEY]
            if metadata[DESCRIPTION_LEN] != description_len:
                print(
                    f"Expecting size {metadata[DESCRIPTION_LEN]} but got {description_len}"
                )
                embeddings = self.build_chunk_embeddings(documents)
                if embeddings is None:
                    raise ValueError("No Embeddings were created")
                return embeddings

            self.chunk_embeddings = np.load(file=self.chunk_embeddings_path)

            return self.chunk_embeddings
        else:
            embeddings = self.build_chunk_embeddings(documents)
            if embeddings is None:
                raise ValueError("No embeddings were created")
            return embeddings

    def search_chunks(self, query: str, limit: int = 10):
        q_embedding = self.generate_embedding(query)
        chunk_score = []

        if self.chunk_embeddings is None:
            raise ValueError("No embeddings present build embeddings first")
        if self.chunk_metadata is None:
            raise ValueError("No metadata present build embeddings first")

        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(q_embedding, chunk_embedding)
            chunk_score.append(
                {
                    "score": score,
                    CHUNK_INDEX: self.chunk_metadata[index][CHUNK_INDEX],
                    MOVIE_INDEX: self.chunk_metadata[index][MOVIE_INDEX],
                }
            )

        movie_score = {}
        for score in chunk_score:
            if (
                score[MOVIE_INDEX] not in movie_score
                or score["score"] > movie_score[score[MOVIE_INDEX]]
            ):
                movie_score[score[MOVIE_INDEX]] = score["score"]

        movie_score_list = sorted(movie_score.items(), key=lambda x: x[1], reverse=True)
        limit = min(limit, len(movie_score_list))

        res = []
        for movie_index, score in movie_score_list[:limit]:
            movie = self.documents[movie_index]
            res.append(
                {
                    DOCUMENT_KEY: {
                        "id": movie["id"],
                        TITLE_KEY: movie[TITLE_KEY],
                        DESCRIPTION_KEY: movie[DESCRIPTION_KEY][:100],
                    },
                    "score": round(score, SCORE_PRECISION),
                    "metadata": {},
                }
            )
        return res


def embed_chunks() -> np.ndarray:
    documents = import_json()
    ch_sem_search = ChunkedSemanticSearch()
    return ch_sem_search.load_or_create_embeddings(documents)


def search_chunked(query: str, limit) -> list[dict]:
    ch_sem_search = ChunkedSemanticSearch()
    documents = import_json()
    ch_sem_search.load_or_create_embeddings(documents)
    return ch_sem_search.search_chunks(query, limit)
