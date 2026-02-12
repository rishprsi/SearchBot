import argparse

from dotenv import load_dotenv

from lib.constants import DOCUMENT_KEY, QUERY_KEY, RELEVANT_DOCS, TITLE_KEY
from lib.hybrid_search import rrf_search
from lib.search_utils import import_golden_dataset


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results ot evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    testcases = import_golden_dataset()
    load_dotenv()
    for testcase in testcases:
        results = rrf_search(testcase[QUERY_KEY], 60, limit, "", "", False)
        relevant = 0

        result_titles = []
        for result in results:
            if result[DOCUMENT_KEY][TITLE_KEY] in testcase[RELEVANT_DOCS]:
                relevant += 1
            result_titles.append(result[DOCUMENT_KEY][TITLE_KEY])
        precision = relevant / len(results)
        relevance = relevant / len(testcase[RELEVANT_DOCS])
        f1_score = 2 * (precision * relevance) / (precision + relevance)
        print(f"- Query: {testcase[QUERY_KEY]}")
        print(f"- Precision@{len(results)}: {precision:.4f}")
        print(f"- Recall@{len(results)}: {relevance:.4f}")
        print(f"- F1 Score: {f1_score:.4f}")
        print(f"Retrieved: {', '.join(result_titles)}")
        print(f"Relevant: {', '.join(testcase[RELEVANT_DOCS])}")


if __name__ == "__main__":
    main()
