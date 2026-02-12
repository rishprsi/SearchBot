import argparse
from dotenv import load_dotenv

from lib.constants import (
    BM25_RANK,
    BM25_SCORE,
    DESCRIPTION_KEY,
    DOCUMENT_KEY,
    HYBRID_SCORE,
    LLM_SCORE,
    RRF_SCORE,
    SEM_RANK,
    SEM_SCORE,
)
from lib.hybrid_search import rrf_search, weighted_search

commands = {
    "normalize": ["scores"],
    "weighted-search": ["query"],
    "rrf-search": ["query"],
}

opt_args = {
    "weighted-search": [("alpha", 0.5), ("limit", 5)],
    "rrf-search": [("k", 60), ("limit", 5)],
}

bool_args = {"rrf-search": [("evaluate", "store_true")]}

choice_args = {
    "rrf-search": {
        "enhance": ["spell", "rewrite", "expand"],
        "rerank-method": ["individual", "batch", "cross_encoder"],
    }
}

query_type = {
    "query": str,
    "doc_id": int,
    "term": str,
    "text": str,
    "alpha": float,
    "k": int,
    "limit": int,
    "chunk-size": int,
    "overlap": int,
    "max-chunk-size": int,
    "scores": float,
    "enhance": str,
    "rerank-method": str,
    "evaluate": bool,
}

help = {
    # A list of commands
    "normalize": "Normalize a list of scores to give value between 0 to 1",
    "weighted-search": "Weighted search with weighted keyword and semantic results",
    "rrf-search": "Search for a movie using Reciprocal Rank Fusion",
    # Query help
    "scores": "A list of scores you want to normalize",
    "query": "Query text that you want to search for",
    # Optional query argument
    "alpha": "0-1, The greater the value the more weightage keyword search will have",
    "limit": "The number of results returned",
    "k": "Constant to control the weight of higher-ranked results",
    "evaluate": "Evaluate the results using an LLM",
    # Choice arguments
    "enhance": "Enhance the search with the following choices: spell, rewrite, expand",
    "rerank-method": "Reranking hight ranking documents by document analysis ",
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
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
        if command in bool_args:
            for bool_arg, action in bool_args[command]:
                command_parsers[command].add_argument(
                    "--" + bool_arg, action=action, help=help[bool_arg]
                )
        if command in choice_args:
            for choice_arg in choice_args[command].keys():
                command_parsers[command].add_argument(
                    "--" + choice_arg,
                    type=query_type[choice_arg],
                    choices=choice_args[command][choice_arg],
                    help=help[choice_arg],
                )
    args = parser.parse_args()
    match args.command:
        case "normalize":
            print("Normalizing scores")
            scores = args.scores
            if not scores:
                return
            minimum = min(scores)
            maximum = max(scores)
            if maximum == minimum:
                print(1.0)
                return
            for score in scores:
                print((score - minimum) / (maximum - minimum))
        case "weighted-search":
            print("Performing weighted search")
            results = weighted_search(args.query, args.alpha, args.limit)
            for index, result in enumerate(results):
                doc = result[DOCUMENT_KEY]
                print(f"{index + 1}. {doc['title']}")
                print(f"    Hybrid Score: {result[HYBRID_SCORE]:.3f}")
                print(
                    f"    BM25: {result[BM25_SCORE]:.3f}, Semantic: {result[SEM_SCORE]:.3f}"
                )
                print(f"    {doc[DESCRIPTION_KEY]}")
        case "rrf-search":
            print("Performing Reciprocal Rank Fusion Search")
            results = rrf_search(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )
            for index, result in enumerate(results):
                doc = result[DOCUMENT_KEY]
                print(f"{index + 1}. {doc['title']}")
                if args.evaluate:
                    print(f"    LLM Evaluation Scores:{result.get(LLM_SCORE, 0)}/3")
                if args.rerank_method and result.get(RRF_SCORE):
                    print(f"    Rerank Score: {result[RRF_SCORE]}")
                print(f"    RRF Score: {result[HYBRID_SCORE]:.3f}")
                print(
                    f"    BM25 Rank: {result[BM25_RANK]:.3f}, Semantic Rank: {result[SEM_RANK]:.3f}"
                )
                print(f"    {doc[DESCRIPTION_KEY]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
