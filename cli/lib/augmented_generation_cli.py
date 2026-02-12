import os
from google import genai

from .constants import DESCRIPTION_KEY, DOCUMENT_KEY, TITLE_KEY
from .hybrid_search import rrf_search


MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def get_llm_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("Please provide an api key before proceeding")

    return genai.Client(api_key=api_key)


def rag(query):
    results = rrf_search(query, 60, 5, "", "", False, True)
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


def summarize(query, limit):
    results = rrf_search(query, 60, limit, "", "", False, True)
    print("Search Results:")
    for result in results:
        print(f"    - {result[DOCUMENT_KEY][TITLE_KEY]}")
    formatted_results = [
        f"Title: {result[DOCUMENT_KEY][TITLE_KEY]}, Description: {result[DOCUMENT_KEY][DESCRIPTION_KEY]}"
        for result in results
    ]
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{formatted_results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text is not None:
        print("LLM Summary:")
        print(content.text)


def citations(query, limit):
    results = rrf_search(query, 60, limit, "", "", False)
    print("Search Results:")
    for result in results:
        print(f"    - {result[DOCUMENT_KEY][TITLE_KEY]}")

    formatted_results = [
        f"Title: {result[DOCUMENT_KEY][TITLE_KEY]}, Description: {result[DOCUMENT_KEY][DESCRIPTION_KEY]}"
        for result in results
    ]

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{formatted_results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text is not None:
        print("LLM Answer:")
        print(content.text)


def questions(question, limit):
    results = rrf_search(question, 60, limit, "", "", False, True)
    print("Search Results:")
    for result in results:
        print(f"    - {result[DOCUMENT_KEY][TITLE_KEY]}")

    formatted_results = [
        f"Title: {result[DOCUMENT_KEY][TITLE_KEY]}, Description: {result[DOCUMENT_KEY][DESCRIPTION_KEY]}"
        for result in results
    ]

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{formatted_results}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text is not None:
        print("Answer:")
        print(content.text)
