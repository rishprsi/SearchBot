import argparse
import traceback
from lib.constants import TITLE_KEY
from lib.keyword_search import (
    search_title,
    get_idf,
    get_tf,
    build,
    get_tfidf,
    get_bm25_idf,
    get_bm25_tf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser("build", help="Build cache")

    # Adding subparrsers for tf document id and term
    tf_parser = subparsers.add_parser(
        "tf", help="Check term frequency for a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID for the term")
    tf_parser.add_argument("term", type=str, help="Term to search for in the document")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Check term frequency for a term in a document"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID for the term")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to search for in the document"
    )

    # Adding inverse frequency command
    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a term"
    )
    idf_parser.add_argument("term", help="Term's Inverse document frequency")

    idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF for a term")
    idf_parser.add_argument("term", help="Term to get the IDF of")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get tfidf for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID for the term")
    tfidf_parser.add_argument(
        "term", type=str, help="Term to search for in the document"
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
