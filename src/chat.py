
from retrieve import load_chunks, load_or_build_index, search
from generate import generate_answer
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import faiss
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = PROJECT_ROOT / "data" / "chunks" / "faiss.index"


def retrieve_context(question, model, index, chunks, top_k=3):
    code_match = re.search(
        r"\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b",
        question,
        re.IGNORECASE
    )

    if code_match:
        requested_code = re.sub(
            r"\s+", " ", code_match.group(1)
        ).strip().upper()

        matching_chunks = [
            chunk for chunk in chunks
            if re.sub(
                r"\s+", " ", chunk.get("course_code", "")
            ).strip().upper() == requested_code
        ]

        if matching_chunks:
            unique_chunks = []
            seen = set()

            for chunk in matching_chunks:
                normalized_text = re.sub(
                    r"\s+", " ", chunk["text"]
                ).strip().lower()

                if normalized_text not in seen:
                    seen.add(normalized_text)
                    unique_chunks.append(chunk)

            return unique_chunks[:top_k]

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, index.ntotal)
    )

    results = []
    for idx in indices[0]:
        if idx >= 0:
            results.append(chunks[idx])

    return results


def main():
    chunks = load_chunks()

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    # Load the index built by retrieve.py.
    index = faiss.read_index(str(INDEX_FILE))

    while True:
        question = input(
            "\nAsk UniQuery AI (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Chat ended.")
            break

        if not question:
            continue

        results = retrieve_context(
            question,
            embedding_model,
            index,
            chunks
        )

        if not results:
            print(
                "\nUniQuery AI: I could not find relevant "
                "information in the university documents."
            )
            continue

        context_parts = []

        for chunk in results:
            context_parts.append(
                f"Course: {chunk.get('course_code', 'N/A')} "
                f"{chunk.get('course_title', '')}\n"
                f"Source: {chunk['source']}, "
                f"PDF page: {chunk['page']}\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        try:
            answer = generate_answer(question, context)
            print("\nUniQuery AI:", answer)

            print("\nSources:")
            for chunk in results:
                print(
                    f"- {chunk['source']}, "
                    f"PDF page {chunk['page']}"
                )

        except Exception as error:
            print("\nCould not generate an answer:", error)


if __name__ == "__main__":
    main()
