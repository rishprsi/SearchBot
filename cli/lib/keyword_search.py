from .search_utils import import_json
from .preprocess import preprocess


DEFAULT_SEARCH_LIMIT = 5


def search_title(keyword, limit=DEFAULT_SEARCH_LIMIT):
    filtered_movies = []
    movies = import_json()
    keyword_tokens = preprocess(keyword)
    for movie in movies:
        if "title" not in movie:
            print(movie)
            print("\n\n")
            continue
        title_tokens = preprocess(movie["title"])
        if has_matching_token(keyword_tokens, title_tokens):
            filtered_movies.append(movie)
            if len(filtered_movies) >= DEFAULT_SEARCH_LIMIT:
                break

    # filtered_movies.sort(key=lambda x: x["id"])
    length = len(filtered_movies)
    for index, movie in enumerate(filtered_movies):
        print(f"{index + 1}. {movie['title']}")


def has_matching_token(keyword_tokens: list[str], title_tokens: list[str]) -> bool:
    for keyword in keyword_tokens:
        for title_token in title_tokens:
            if keyword in title_token:
                # print(f"Checking token {title_token} and {keyword}")
                return True
    return False
