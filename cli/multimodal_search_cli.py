import argparse
from dotenv import load_dotenv

from lib.multimodal_search import search_with_image, verify_image_embedding
from lib.argparse_util import get_parser


commands = {"verify_image_embedding": ["image_path"], "image_search": ["image_path"]}

opt_args = {}

bool_args = {}

choice_args = {}

query_type = {"image_path": str}

help = {
    # A list of commands
    "verify_image_embedding": "Get the embedding for the provided image",
    "image_search": "Search movie by image",
    # Query help
    "image_path": "Path of the image for the embedding",
    # Optional query argument
    # Choice arguments
}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    get_parser(parser, commands, opt_args, choice_args, bool_args, query_type, help)
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            search_with_image(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
