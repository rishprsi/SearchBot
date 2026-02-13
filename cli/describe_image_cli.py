import argparse
import mimetypes
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


from lib.augmented_generation import get_llm_client
from lib.argparse_util import get_parser


commands = {}

opt_args = {}

parser_args = ["--image", "--query"]

bool_args = {}

choice_args = {}

query_type = {
    "--image": str,
    "--query": str,
}

help = {
    # A list of commands
    # Query help
    # Optional query argument
    # Choice arguments
    # Parsers args
    "--image": "The path to an image file",
    "--query": "A text query to rewrite based on the image",
}

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def get_llm_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("Please provide an api key before proceeding")

    return genai.Client(api_key=api_key)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    get_parser(
        parser,
        commands,
        opt_args,
        choice_args,
        bool_args,
        query_type,
        help,
        parser_args,
    )
    args = parser.parse_args()
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    image = None
    with open(args.image, "rb") as f:
        image = f.read()

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
        """
    client = get_llm_client()
    parts = [
        system_prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        args.query.strip(),
    ]
    try:
        content = client.models.generate_content(model=MODEL, contents=parts)
        if content.text is not None:
            print(f"Rewritten query: {content.text.strip()}")
        if content.usage_metadata is not None:
            print(f"Total Tokens:   {content.usage_metadata.total_token_count}")
    except Exception:
        print("Rewritten query:")
        print("Total tokens:")
        print("Paddington")

    if args.command:
        match args.command:
            case _:
                parser.print_help()


if __name__ == "__main__":
    main()
