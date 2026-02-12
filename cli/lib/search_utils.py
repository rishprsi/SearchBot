import json
from .constants import (
    GOLDEN_DATA_PATH,
    MOVIEKEY,
    DATA_PATH,
    STOPWORD_PATH,
    TESTCASE_KEY,
)
import regex as re


def import_json():
    json_dict = {}
    with open(DATA_PATH, "r") as f:
        json_dict = json.load(f)

    return json_dict[MOVIEKEY]


def import_golden_dataset():
    json_dict = {}

    with open(GOLDEN_DATA_PATH, "r") as f:
        json_dict = json.load(f)
    return json_dict[TESTCASE_KEY]


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
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    length = len(sentences)

    for index in range(length):
        curr = sentences[index]
        curr = curr.strip()
        if not curr:
            sentences.pop(index)
        sentences[index] = curr
    res = [" ".join(sentences[: min(max_chunk_size, length)])]
    curr = max_chunk_size
    increments = max_chunk_size - overlap

    while curr < length:
        res.append(" ".join(sentences[curr - overlap : min(length, curr + increments)]))
        curr += increments

    return res
