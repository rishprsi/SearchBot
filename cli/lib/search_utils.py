import json
from .constants import MOVIEKEY, DATA_PATH, STOPWORD_PATH


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
