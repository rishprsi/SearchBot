import argparse
from dotenv import load_dotenv

from cli.lib.argparse_util import get_parser
from lib.augmented_generation_cli import rag


commands = {"rag": ["query"]}

opt_args = {}

bool_args = {}

choice_args = {}

query_type = {"query": str}

help = {
    # A list of commands
    "rag": "Perform RAG (search + generate answer)",
    # Query help
    "query": "Search query for RAG",
    # Optional query argument
    # Choice arguments
}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    get_parser(parser, commands, opt_args, choice_args, bool_args, query_type, help)
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag(query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
