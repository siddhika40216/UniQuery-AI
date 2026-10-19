from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
INDEX_FILE = PROJECT_ROOT / "data" / "chunks" / "faiss.index"
METADATA_FILE = PROJECT_ROOT / "data" / "chunks" / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_search_index(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Creating embeddings for {len(texts)} chunks...")
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

    print(f"\nIndex saved: {INDEX_FILE}")
    print(f"Metadata saved: {METADATA_FILE}")

    return model, index, chunks


def search(query, model, index, chunks, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    print(f"\nQuestion: {query}")
    print("\nMost relevant results:\n")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx < 0:
            continue

        chunk = chunks[idx]

        print(f"Result {rank} | Similarity: {score:.3f}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {chunk['text'][:1500]}")
        print("-" * 60)


def main():
    if not CHUNKS_FILE.exists():
        print("chunks.json not found. Run python src/chunking.py first.")
        return

    chunks = load_chunks()

    if not chunks:
        print("No chunks available to search.")
        return

    model, index, chunks = build_search_index(chunks)

    question = input("\nEnter a university-related question: ").strip()

    if question:
        search(question, model, index, chunks)


if __name__ == "__main__":
    main()
