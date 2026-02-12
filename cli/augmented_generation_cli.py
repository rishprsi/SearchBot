import argparse
from dotenv import load_dotenv

from lib.augmented_generation_cli import citations, questions, summarize
from lib.argparse_util import get_parser
from lib.augmented_generation import rag


commands = {
    "rag": ["query"],
    "summarize": ["query"],
    "citations": ["query"],
    "question": ["question"],
}

opt_args = {
    "summarize": [("limit", 5)],
    "citations": [("limit", 5)],
    "question": [("limit", 5)],
}

bool_args = {}

choice_args = {}

query_type = {"query": str, "limit": int, "question": str}

help = {
    # A list of commands
    "rag": "Perform RAG (search + generate answer)",
    "summarize": "Summarize the results from the search",
    "citations": "Provide summary for the generated content with citations",
    "question": "Ask a question based on provided data",
    # Query help
    "query": "Search query for RAG",
    # Optional query argument
    "limit": "Limits the number of searches that are returned",
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
        case "summarize":
            summarize(args.query, args.limit)
        case "citations":
            citations(args.query, args.limit)
        case "question":
            questions(args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
