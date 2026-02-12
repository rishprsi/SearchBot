import os
from google import genai

from lib.constants import DESCRIPTION_KEY, DOCUMENT_KEY, TITLE_KEY
from .hybrid_search import rrf_search


MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def get_llm_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("Please provide an api key before proceeding")

    return genai.Client(api_key=api_key)


def rag(query):
    results = rrf_search(query, 60, 5, "", "", False)
    result_docs = [
        f"Title: {result[DOCUMENT_KEY][TITLE_KEY]} - Description: {result[DOCUMENT_KEY][DESCRIPTION_KEY]}"
        for result in results
    ]
    prompt = (
        prompt
    ) = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{chr(10).join(result_docs)}

Provide a comprehensive answer that addresses the query:"""

    print("Search Results: ")
    for result in results:
        print(f"    - {result[DOCUMENT_KEY][TITLE_KEY]}")

    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text is not None:
        print("RAG Response:")
        print(content.text)
