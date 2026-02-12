import argparse
from dotenv import load_dotenv

from lib.augmented_generation_cli import rag


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag(query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
