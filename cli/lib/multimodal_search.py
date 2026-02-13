from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import import_json
from .constants import SCORE_KEY, TITLE_KEY, DESCRIPTION_KEY

import numpy as np


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.texts = [
            f"{document[TITLE_KEY]}: {document[DESCRIPTION_KEY]}"
            for document in documents
        ]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        print(self.text_embeddings[0].shape)

    def embed_image(self, image_path: str):
        # with open(image_path, "rb") as f:
        #     image = Image.open(f)
        #     f.close()

        image = Image.open(image_path)
        embeddings = self.model.encode([image.convert("RGB")], show_progress_bar=True)
        return embeddings

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        if len(image_embedding) <= 0:
            return []
        results = []
        for index, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            print("Similarity is: ", similarity)
            text_split = self.texts[index].split(": ")
            title = text_split[0]
            description = ": ".join(text_split[1:])
            results.append(
                {
                    TITLE_KEY: title,
                    DESCRIPTION_KEY: description,
                    SCORE_KEY: similarity[0],
                }
            )
        results.sort(key=lambda x: x[SCORE_KEY], reverse=True)
        return results


def verify_image_embedding(image_path):
    documents = import_json()
    multimodal_search = MultimodalSearch(documents)
    embeddings = multimodal_search.embed_image(image_path)
    embedding = embeddings[0]
    print(f"Embedding shape: {embedding.shape} dimensions")


def search_with_image(image_path):
    documents = import_json()
    multimodal_search = MultimodalSearch(documents)
    results = multimodal_search.search_with_image(image_path)
    for index, i in enumerate(range(5)):
        print(
            f"{index}. {results[index][TITLE_KEY]} (similarity: {results[index][SCORE_KEY]:.3f})"
        )
        print(results[index][DESCRIPTION_KEY])
        print("\n")


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
