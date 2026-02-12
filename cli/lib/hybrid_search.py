import json
import os
from google import genai
from time import sleep
import traceback
from sentence_transformers import CrossEncoder

from .search_utils import import_json
from .constants import (
    BM25_RANK,
    BM25_SCORE,
    DESCRIPTION_KEY,
    DOCUMENT_KEY,
    HYBRID_SCORE,
    LLM_SCORE,
    RRF_SCORE,
    SCORE_KEY,
    SEM_RANK,
    SEM_SCORE,
    TITLE_KEY,
)

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
CROSS_ENCODER = "cross-encoder/ms-marco-TinyBERT-L2-v2"


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_res = self._bm25_search(query, limit * 500)
        sem_res = self.semantic_search.search_chunks(query, limit * 500)
        normalized_bm25 = normalize([x[SCORE_KEY] for x in bm25_res])
        normalized_sem = normalize([x[SCORE_KEY] for x in sem_res])
        final = {}

        final = score_dict(bm25_res, normalized_bm25, final, BM25_SCORE)
        final = score_dict(sem_res, normalized_sem, final, SEM_SCORE)

        for key in final.keys():
            final[key][HYBRID_SCORE] = hybrid_score(
                final[key][BM25_SCORE], final[key][SEM_SCORE], alpha
            )
            final[key][DOCUMENT_KEY][DESCRIPTION_KEY] = final[key][DOCUMENT_KEY][
                DESCRIPTION_KEY
            ][:100]

        sorted_values = sorted(
            final.values(), key=lambda x: x[HYBRID_SCORE], reverse=True
        )
        limit = min(limit, len(sorted_values))
        return sorted_values[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_res = self._bm25_search(query, limit * 500)
        sem_res = self.semantic_search.search_chunks(query, limit * 500)
        final = {}

        rank_dict(bm25_res, final, BM25_RANK)
        rank_dict(sem_res, final, SEM_RANK)

        for key in final.keys():
            bm_rrf = rrf_score(final[key][BM25_RANK], k) if final[key][BM25_RANK] else 0
            sem_rrf = rrf_score(final[key][SEM_RANK], k) if final[key][SEM_RANK] else 0
            final[key][HYBRID_SCORE] = bm_rrf + sem_rrf
            final[key][DOCUMENT_KEY][DESCRIPTION_KEY] = final[key][DOCUMENT_KEY][
                DESCRIPTION_KEY
            ]

        sorted_values = sorted(
            final.values(), key=lambda x: x[HYBRID_SCORE], reverse=True
        )
        limit = min(limit * 5, len(sorted_values))
        return sorted_values[:limit]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def weighted_search(query, alpha, limit):
    documents = import_json()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.weighted_search(query, alpha, limit)


def rrf_search(query, k, limit, enhance, rerank_method, evaluate):
    if enhance == "spell":
        query = llm_spellcheck(query)
    elif enhance == "rewrite":
        query = llm_rewrite(query)
    elif enhance == "expand":
        query = llm_expand(query)
    documents = import_json()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit)
    if rerank_method == "individual":
        try:
            for index, result in enumerate(results):
                results[index][RRF_SCORE] = llm_rerank_individual(
                    query, result[DOCUMENT_KEY], result[HYBRID_SCORE]
                )
            results.sort(key=lambda x: x[RRF_SCORE])
        except Exception:
            traceback.print_exc()
    elif rerank_method == "batch":
        try:
            results = llm_rerank_batch(query, results)
        except Exception:
            traceback.print_exc()
    elif rerank_method == "cross_encoder":
        results = cross_encoder_rerank(query, results)

    llm_scores = []
    limit = min(limit, len(results))
    if evaluate:
        llm_scores = llm_evaluate_results(query, results[:limit])
    for index, result in enumerate(results):
        results[index][DOCUMENT_KEY][DESCRIPTION_KEY] = results[index][DOCUMENT_KEY][
            DESCRIPTION_KEY
        ][:100]

    for index, llm_score in enumerate(llm_scores):
        results[index][LLM_SCORE] = llm_score

    return results[:limit]


def normalize(scores: list):
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return scores
    den = maximum - minimum
    for index in range(len(scores)):
        scores[index] = (scores[index] - minimum) / den
    return scores


def score_dict(init_dicts, normalized_scores, final, key):
    for index, init_dict in enumerate(init_dicts):
        doc = init_dict[DOCUMENT_KEY]
        if doc["id"] not in final:
            final[doc["id"]] = {
                DOCUMENT_KEY: doc,
                BM25_SCORE: 0.0,
                SEM_SCORE: 0.0,
                # HYBRID_SCORE: 0.0,
            }

        final[doc["id"]][key] = normalized_scores[index]

    return final


def rank_dict(init_dicts, final, key):
    init_dicts.sort(key=lambda x: x[SCORE_KEY], reverse=True)
    for index, init_dict in enumerate(init_dicts):
        doc = init_dict[DOCUMENT_KEY]
        if doc["id"] not in final:
            final[doc["id"]] = {
                DOCUMENT_KEY: doc,
                BM25_RANK: 0,
                SEM_RANK: 0,
            }
        final[doc["id"]][key] = index + 1

    return final


def get_llm_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("Please provide an api key before proceeding")

    return genai.Client(api_key=api_key)


def llm_spellcheck(query):
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text != query:
        print(f"Enhanced query (spell): '{query}' -> '{content.text}'\n")

    return content.text


def llm_rewrite(query):
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text != query:
        print(f"Enhanced query (rewrite): '{query}' -> '{content.text}'\n")

    return content.text


def llm_expand(query):
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text != query:
        print(f"Enhanced query (expand): '{query}' -> '{content.text}'\n")

    return content.text


def llm_rerank_individual(query, doc, score):
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text is not None:
        score = float(content.text)

    sleep(3)

    return score


def llm_rerank_batch(query, results):
    results_dict = {}
    doc_strings = []
    for result in results:
        doc_strings.append(
            f"ID: {result[DOCUMENT_KEY]['id']}, Title: {result[DOCUMENT_KEY][TITLE_KEY]}, Description: {result[DOCUMENT_KEY][DESCRIPTION_KEY]}"
        )
        results_dict[result[DOCUMENT_KEY]["id"]] = result

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_strings}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    print(f"Model is {MODEL}")
    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    if content.text is None:
        raise ValueError("No response from LLM")
    response_indexes = json.loads(content.text)
    new_results = []
    for index in response_indexes:
        new_results.append(results_dict[index])
    return new_results


def cross_encoder_rerank(query, results):
    pairs = []
    for result in results:
        doc = result[DOCUMENT_KEY]
        # print(f"{doc.get(TITLE_KEY, '')} - {doc.get(DESCRIPTION_KEY, '')}")
        pairs.append(
            [query, f"{doc.get(TITLE_KEY, '')} - {doc.get(DESCRIPTION_KEY, '')}"]
        )
    cross_encoder = CrossEncoder(CROSS_ENCODER)
    scores = cross_encoder.predict(pairs)
    for index, result in enumerate(results):
        results[index][RRF_SCORE] = scores[index]

    results.sort(key=lambda x: x[RRF_SCORE], reverse=True)
    return results


def llm_evaluate_results(query, results):
    formatted_results = [
        f"{result[DOCUMENT_KEY][TITLE_KEY]} - {result[DOCUMENT_KEY][DESCRIPTION_KEY]}"
        for result in results
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    # print("Prompt is: ", prompt)

    client = get_llm_client()
    content = client.models.generate_content(model=MODEL, contents=prompt)
    result = []
    if content.text is not None:
        # print("Getting the content results as: ", content.text)
        result = json.loads(content.text)

    print("Scores from the LLM are: ", result)
    return result
