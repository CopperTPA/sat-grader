
"""
Build the SAT Book Index
------------------------
Reads all extracted JSON books from /books,
cleans and chunks their text, and saves the
result to /rag/chunks.json.

Expected input format:
[
    {"page": 1, "text": "Page text..."},
    {"page": 2, "text": "Page text..."}
]
"""

import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BOOKS_DIR = ROOT_DIR / "books"
RAG_DIR = ROOT_DIR / "rag"
OUTPUT_FILE = RAG_DIR / "chunks.json"

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


def clean_text(text):
    """Normalize extracted PDF text without removing useful content."""
    text = text.replace("\x00", " ")
    # Remove the recurring garbled PDF header observed in the extracted book text.
    text = text.replace("0V ÈQK 'ң\\ 6$7", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text):
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_book(json_path):
    """Read one extracted JSON book."""
    with open(json_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    if not isinstance(pages, list):
        raise ValueError(f"Unexpected JSON structure in {json_path.name}")

    return pages


def build_index():
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    book_files = sorted(BOOKS_DIR.glob("*.json"))

    if not book_files:
        raise FileNotFoundError(
            f"No JSON books found in: {BOOKS_DIR}"
        )

    all_chunks = []

    for book_path in book_files:
        print(f"\nProcessing: {book_path.name}")

        pages = load_book(book_path)
        book_chunk_count = 0

        for page_data in pages:
            if not isinstance(page_data, dict):
                continue

            page_number = page_data.get("page")
            page_text = page_data.get("text", "")

            if not isinstance(page_text, str) or not page_text.strip():
                continue

            cleaned = clean_text(page_text)
            page_chunks = split_into_chunks(cleaned)

            for chunk_number, chunk_text in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "id": len(all_chunks),
                    "book": book_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk_text
                })

                book_chunk_count += 1

        print(f"Pages processed: {len(pages)}")
        print(f"Chunks generated: {book_chunk_count}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            separators=(",", ":")
        )

    print("\nIndex preparation completed!")
    print(f"Books processed: {len(book_files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_index()
