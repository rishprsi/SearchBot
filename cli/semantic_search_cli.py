#!/usr/bin/env python3

import argparse

from cli.lib.argparse_util import get_parser
from lib.constants import DESCRIPTION_KEY, DOCUMENT_KEY, TITLE_KEY
from lib.chunked_semantic_search import embed_chunks, search_chunked
from lib.search_utils import chunk_text, semantic_chunk_text
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


commands = {
    "search": ["query"],
    "search_chunked": ["query"],
    "verify": [],
    "embed_text": ["text"],
    "verify_embeddings": [],
    "embedquery": ["query"],
    "chunk": ["text"],
    "semantic_chunk": ["text"],
    "embed_chunks": [],
}

opt_args = {
    "search": [("limit", 5)],
    "search_chunked": [("limit", 5)],
    "chunk": [("chunk-size", 200), ("overlap", 20)],
    "semantic_chunk": [("max-chunk-size", 4), ("overlap", 0)],
}

query_type = {
    "query": str,
    "doc_id": int,
    "term": str,
    "text": str,
    "limit": int,
    "chunk-size": int,
    "overlap": int,
    "max-chunk-size": int,
}

help = {
    # Help for commands
    "search": "Search a term from the database",
    "search_chunked": "Search a term after chunking the database",
    "verify": "Verifies that the model is up and functional",
    "embed_text": "Embeds a  text into vector embeddings for semantic search",
    "verify_embeddings": "Builds the embeddings / loads from cache if it already exists",
    "embedquery": "Get the embedding of a query text",
    "chunk": "Chunk the provided text into fixed size chunks",
    "semantic_chunk": "Convert text into semantic chunks",
    "embed_chunks": "Embed chunks from the documents and cache it",
    # Help for arguments
    "query": "The term you need to search for",
    "text": "Text input to be processed",
    # Positional arguments
    "limit": "Number of results as output (Default 5)",
    "chunk-size": "The size of each chunk (Default 200)",
    "overlap": "Number of words to overlap in chunks (Default 0)",
    "max-chunk-size": "Maximum allowed size of the chunk (Default 4)",
}


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser = get_parser(parser, commands, opt_args, {}, {}, query_type, help)
    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, args.limit)
        case "search_chunked":
            movies = search_chunked(args.query, args.limit)
            for index, movie_dict in enumerate(movies):
                movie = movie_dict[DOCUMENT_KEY]
                print(
                    f"\n{index + 1}. {movie[TITLE_KEY]} (score: {movie_dict['score']:.4f})"
                )
                print(f"{movie[DESCRIPTION_KEY]}...")
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
        case "semantic_chunk":
            sentences = semantic_chunk_text(
                args.text, args.max_chunk_size, args.overlap
            )
            print(f"Semantically chunking {len(args.text)} characters")
            for index, sentence in enumerate(sentences):
                print(f"{index + 1}. {sentence}")
        case "embed_chunks":
            embeddings = embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
