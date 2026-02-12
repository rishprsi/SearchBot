import argparse
import traceback
from lib.constants import DOCUMENT_KEY, SCORE_KEY, TITLE_KEY
from lib.keyword_search import (
    bm25_search_title,
    search_title,
    get_idf,
    get_tf,
    build,
    get_tfidf,
    get_bm25_idf,
    get_bm25_tf,
)

commands = {
    "search": ["query"],
    "bm25search": ["query"],
    "build": [],
    "tf": ["doc_id", "term"],
    "bm25tf": ["doc_id", "term"],
    "idf": ["term"],
    "bm25idf": ["term"],
    "tfidf": ["doc_id", "term"],
}

query_type = {"query": str, "doc_id": int, "term": str}

help = {
    # Command help
    "search": "Search movies using BM25",
    "bm25search": "Improved search with bm25",
    "build": "Build cache",
    "tf": "Check term frequency for a term in a document",
    "bm25tf": "Check term frequency for a term in a document",
    "idf": "Get inverse document frequency for a term",
    "bm25idf": "Get BM25 IDF for a term",
    "tfidf": "Get tfidf for a term in a document",
    # Argument help
    "query": "Search query",
    "doc_id": "Document ID for the term",
    "term": "Term to search for in the document",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parsers = dict()

    for command in commands.keys():
        parsers[command] = subparsers.add_parser(command, help=help[command])
        for argument in commands[command]:
            parsers[command].add_argument(
                argument, type=query_type[argument], help=help[argument]
            )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                filtered_movies = search_title(args.query)
                for index, movie in enumerate(filtered_movies):
                    print(f"{index + 1}. {movie[TITLE_KEY]}")
            except Exception:
                traceback.print_exc()
        case "bm25search":
            print(f"Searching for: {args.query} using bm25")
            filtered_movies = bm25_search_title(args.query)
            for index, movie_dict in enumerate(filtered_movies):
                movie, score = movie_dict[DOCUMENT_KEY], movie_dict[SCORE_KEY]
                print(
                    f"{index + 1}. ({movie['id']}) {movie[TITLE_KEY]} - Score: {score:.2f}"
                )
        case "build":
            print("Building pickle files")
            build()
        case "tf":
            print(f"Fetching term frequency from {args.doc_id} of {args.term}")
            count = get_tf(args.doc_id, args.term)
            print(f"The frequency of the term is {count}")
        case "bm25tf":
            print(f"Checking the BM25 TF for {args.term}")
            idf = get_bm25_tf(args.doc_id, args.term)
            print(f"The BM25TF of the term {args.term} is {idf:.2f}")
        case "idf":
            print(f"Checking the inverse frequency for {args.term}")
            idf = get_idf(args.term)
            print(f"The IDF of the term {args.term} is {idf:.2f}")
        case "bm25idf":
            print(f"Checking the BM25 IDF for {args.term}")
            idf = get_bm25_idf(args.term)
            print(f"The BM25IDF of the term {args.term} is {idf:.2f}")
        case "tfidf":
            print(f"Getting tfidf for the term {args.term}")
            tfidf = get_tfidf(args.doc_id, args.term)
            print(
                f"The TF-IDF for term {args.term} in doc_id {args.doc_id} is {tfidf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
