import json
import os

MOVIEKEY = "movies"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


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
