
"""
SAT Book Retriever
------------------
Loads chunks.json, builds a BM25 search index,
and retrieves relevant passages from both books.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT_DIR / "rag" / "chunks.json"

DEFAULT_TOP_K = 5


def tokenize(text):
    """Convert text into searchable tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class SATBookRetriever:

    def __init__(self, chunks_file=CHUNKS_FILE):
        self.chunks_file = Path(chunks_file)
        self.chunks = []
        self.bm25 = None

        self.load_index()

    def load_index(self):
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_file}. "
                "Run rag/build_index.py first."
            )

        with open(self.chunks_file, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if not self.chunks:
            raise ValueError("The chunks file is empty.")

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(
            f"BM25 loaded successfully: "
            f"{len(self.chunks)} chunks."
        )

    def search(self, query, top_k=DEFAULT_TOP_K, book=None):
        """
        Search the books.

        Args:
            query: SAT question or extracted topic.
            top_k: Maximum number of passages to return.
            book: Optional exact filename, e.g. book1_extracted.json.

        Returns:
            List of relevant passages with page references and scores.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        results = []

        for index, score in enumerate(scores):
            chunk = self.chunks[index]

            if book and chunk["book"] != book:
                continue

            if score <= 0:
                continue

            results.append({
                "book": chunk["book"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "text": chunk["text"],
                "score": float(score)
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:max(1, int(top_k))]

    def format_context(self, results):
        """Format retrieved passages for Claude."""
        if not results:
            return "No relevant passages were found in the SAT books."

        formatted = []

        for i, result in enumerate(results, start=1):
            formatted.append(
                f"[Source {i}]\n"
                f"Book: {result['book']}\n"
                f"Page: {result['page']}\n"
                f"Passage:\n{result['text']}"
            )

        return "\n\n".join(formatted)


if __name__ == "__main__":
    retriever = SATBookRetriever()

    test_query = (
        "subject verb agreement singular plural "
        "verbs grammar"
    )

    results = retriever.search(test_query, top_k=5)

    print("\nSearch results:\n")

    for result in results:
        print(
            f"{result['book']} | "
            f"Page {result['page']} | "
            f"Score: {result['score']:.3f}"
        )
        print(result["text"][:400])
        print("-" * 60)
