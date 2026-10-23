
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieve import load_chunks
from generate import generate_answer
from sentence_transformers import SentenceTransformer
import faiss
import re

app = FastAPI(title="UniQuery AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_FILE = PROJECT_ROOT / "data" / "chunks" / "faiss.index"

chunks = []
embedding_model = None
index = None


class ChatRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_rag_system():
    global chunks, embedding_model, index

    chunks = load_chunks()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    index = faiss.read_index(str(INDEX_FILE))


def retrieve_context(question, top_k=3):
    code_match = re.search(
        r"\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b",
        question,
        re.IGNORECASE,
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
                text_key = re.sub(
                    r"\s+", " ", chunk["text"]
                ).strip().lower()

                if text_key not in seen:
                    seen.add(text_key)
                    unique_chunks.append(chunk)

            return unique_chunks[:top_k]

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, index.ntotal),
    )

    return [
        chunks[i]
        for i in indices[0]
        if i >= 0
    ]


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    results = retrieve_context(question)

    if not results:
        return {
            "answer": (
                "I could not find this information in the "
                "available university documents."
            ),
            "sources": [],
        }

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
    answer = generate_answer(question, context)

    sources = [
        {
            "file": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in results
    ]

    return {"answer": answer, "sources": sources}
