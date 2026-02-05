import json
from .constants import MOVIEKEY, DATA_PATH, STOPWORD_PATH
import regex as re


def import_json():
    json_dict = {}
    with open(DATA_PATH, "r") as f:
        json_dict = json.load(f)

    return json_dict[MOVIEKEY]


def load_stopwords() -> list[str]:
    try:
        with open(STOPWORD_PATH, "r") as f:
            return f.read().splitlines()
    except Exception as e:
        print(e)
        return []


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    res = []
    length = len(words)
    curr = 0

    while curr < length:
        res.append(
            " ".join(words[max(0, curr - overlap) : min(length, curr + chunk_size)])
        )
        curr += chunk_size

    return res


def semantic_chunk_text(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    print("incoming text is: ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    res = []
    length = len(sentences)
    curr = 0

    while curr < length:
        res.append(
            " ".join(
                sentences[max(0, curr - overlap) : min(length, curr + max_chunk_size)]
            )
        )
        curr += max_chunk_size

    return res
