
from pathlib import Path
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
INDEX_FILE = PROJECT_ROOT / "data" / "chunks" / "faiss.index"
METADATA_FILE = PROJECT_ROOT / "data" / "chunks" / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COURSE_CODE_PATTERN = re.compile(
    r"\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b",
    re.IGNORECASE
)


def normalize_code(code):
    return re.sub(r"\s+", " ", code).strip().upper()


def load_chunks():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_or_build_index(model, chunks):
    # Load old index only if its metadata exactly matches the new chunks.
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            index = faiss.read_index(str(INDEX_FILE))

            with METADATA_FILE.open("r", encoding="utf-8") as file:
                saved_chunks = json.load(file)

            if saved_chunks == chunks and index.ntotal == len(chunks):
                print(f"Loaded {index.ntotal} indexed course records.")
                return index

        except (ValueError, RuntimeError, OSError, json.JSONDecodeError):
            pass

        print("Chunk data changed. Rebuilding FAISS index...")

    print(f"Creating embeddings for {len(chunks)} course records...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False)

    print("New FAISS index and metadata saved.")
    return index


def search(query, model, index, chunks, top_k=5):
    print(f"\nQuestion: {query}")
    print("\nMost relevant results:\n")

    # Check whether the user mentioned an exact course code.
    code_match = COURSE_CODE_PATTERN.search(query)

    if code_match:
        requested_code = normalize_code(code_match.group(1))

        candidates = [
            chunk for chunk in chunks
            if normalize_code(chunk.get("course_code", "")) == requested_code
        ]

        if candidates:
            # Remove repeated copies of identical course text.
            unique_candidates = []
            seen_texts = set()

            for chunk in candidates:
                normalized_text = re.sub(
                    r"\s+", " ", chunk["text"]
                ).strip().lower()

                if normalized_text not in seen_texts:
                    seen_texts.add(normalized_text)
                    unique_candidates.append(chunk)

            candidates = unique_candidates

            # Rank only the matching course records by semantic similarity.
            query_embedding = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]

            candidate_embeddings = model.encode(
                [chunk["text"] for chunk in candidates],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            scores = candidate_embeddings @ query_embedding
            ranked_indices = np.argsort(scores)[::-1][:top_k]

            print(f"Exact course code matched: {requested_code}")

            for rank, idx in enumerate(ranked_indices, start=1):
                chunk = candidates[int(idx)]

                print(f"\nResult {rank} | Similarity: {scores[idx]:.3f}")
                print(
                    f"Course: {chunk['course_code']} "
                    f"{chunk.get('course_title', '')}"
                )
                print(f"Source: {chunk['source']}")
                print(f"PDF page: {chunk['page']}")
                print(f"Text:\n{chunk['text'][:2500]}")
                print("-" * 60)

            return

        print(
            f"No exact course-code match found for {requested_code}. "
            "Showing general semantic results instead.\n"
        )

    # General question: use the FAISS semantic search index.
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx < 0:
            continue

        chunk = chunks[idx]

        print(f"\nResult {rank} | Similarity: {score:.3f}")
        print(
            f"Course: {chunk.get('course_code', 'Not identified')} "
            f"{chunk.get('course_title', '')}"
        )
        print(f"Source: {chunk['source']}")
        print(f"PDF page: {chunk['page']}")
        print(f"Text:\n{chunk['text'][:1500]}")
        print("-" * 60)


def main():
    if not CHUNKS_FILE.exists():
        print("chunks.json not found. Run python src/chunking.py first.")
        return

    chunks = load_chunks()

    if not chunks:
        print("No course records available.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    index = load_or_build_index(model, chunks)

    while True:
        question = input(
            "\nEnter a university question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Search session ended.")
            break

        if question:
            search(question, model, index, chunks)


if __name__ == "__main__":
    main()
