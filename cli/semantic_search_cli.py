#!/usr/bin/env python3

import argparse

from lib.search_utils import chunk_text
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


commands = {
    "search": ["query"],
    "verify": [],
    "embed_text": ["text"],
    "verify_embeddings": [],
    "embedquery": ["query"],
    "chunk": ["text"],
}

opt_args = {"search": [("limit", 5)], "chunk": [("chunk-size", 200), ("overlap", 20)]}

query_type = {
    "query": str,
    "doc_id": int,
    "term": str,
    "text": str,
    "limit": int,
    "chunk-size": int,
    "overlap": int,
}

help = {
    # Help for commands
    "search": "Search a term from the database",
    "verify": "Verifies that the model is up and functional",
    "embed_text": "Embeds a  text into vector embeddings for semantic search",
    "verify_embeddings": "Builds the embeddings / loads from cache if it already exists",
    "embedquery": "Get the embedding of a query text",
    "chunk": "Chunk the provided text into fixed size chunks",
    # Help for arguments
    "query": "The term you need to search for",
    "text": "Text input to be processed",
    # Positional arguments
    "limit": "Number of results as output (Default 5)",
    "chunk-size": "The size of each chunk (Default 200)",
    "overlap": "Number of words to overlap in chunks",
}


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", description="Available commands")
    command_parsers = dict()

    for command in commands:
        command_parsers[command] = subparsers.add_parser(command, help=help[command])
        for argument in commands[command]:
            command_parsers[command].add_argument(
                argument, type=query_type[argument], help=help[argument]
            )
        if command in opt_args:
            for opt_arg, default_size in opt_args[command]:
                command_parsers[command].add_argument(
                    "--" + opt_arg,
                    type=query_type[opt_arg],
                    help=help[opt_arg],
                    default=default_size,
                )
    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, args.limit)
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks):
                print(f"{index + 1}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
