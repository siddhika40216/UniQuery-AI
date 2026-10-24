
from pathlib import Path
import sys
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieve import load_chunks
from generate import generate_answer
from sentence_transformers import SentenceTransformer
import faiss

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

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []



@app.on_event("startup")
def load_rag_system():
    global chunks, embedding_model, index

    chunks = load_chunks()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    index = faiss.read_index(str(INDEX_FILE))


def retrieve_po_definitions(question):
    extracted_file = (
        PROJECT_ROOT
        / "data"
        / "extracted"
        / "PO CO Undergraduate courses_bv.txt"
    )

    if not extracted_file.exists():
        return []

    text = extracted_file.read_text(encoding="utf-8")

    pages = re.split(
        r"---\s*Page\s+(\d+)\s*---",
        text,
        flags=re.IGNORECASE,
    )

    results = []

    pattern = (
        r"\bPO\s*10\s*:\s*"
        r"Environment\s+and\s+sustainability\s*:?\s*"
        r"(.*?)"
        r"(?=\n\s*PO\s*\d+\s*:|\n\s*\n|$)"
    )

    for i in range(1, len(pages), 2):
        page_number = pages[i]
        page_text = pages[i + 1]

        matches = re.finditer(
            pattern,
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in matches:
            definition = re.sub(
                r"\s+",
                " ",
                match.group(1),
            ).strip()

            if definition:
                results.append({
                    "course_code": "PO10",
                    "course_title": "Environment and sustainability",
                    "source": "PO CO Undergraduate courses_bv.txt",
                    "page": int(page_number),
                    "text": (
                        "PO10: Environment and sustainability: "
                        + definition
                    ),
                })

    return results


def retrieve_context(question, top_k=3):
    # First, check for PO10 questions.
    if re.search(r"\bPO\s*10\b", question, re.IGNORECASE):
        po_results = retrieve_po_definitions(question)

        if po_results:
            return po_results[:top_k]

    # Next, check for an exact course code.
    code_match = re.search(
        r"\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b",
        question,
        re.IGNORECASE,
    )

    if code_match:
        requested_code = re.sub(
            r"\s+",
            " ",
            code_match.group(1),
        ).strip().upper()

        matching_chunks = [
            chunk
            for chunk in chunks
            if re.sub(
                r"\s+",
                " ",
                chunk.get("course_code", ""),
            ).strip().upper() == requested_code
        ]

        if matching_chunks:
            unique_chunks = []
            seen = set()

            for chunk in matching_chunks:
                text_key = re.sub(
                    r"\s+",
                    " ",
                    chunk["text"],
                ).strip().lower()

                if text_key not in seen:
                    seen.add(text_key)
                    unique_chunks.append(chunk)

            return unique_chunks[:top_k]

    # Otherwise, use semantic search.
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
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # Convert conversation history into dictionaries.
    history = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.history
    ]

    # Use the previous user question to identify the course
    # when the current question is a follow-up.
    retrieval_question = question

    has_course_code = re.search(
        r"\b[A-Z]{2,}\s+\d{3}[A-Z]?\b",
        question,
        re.IGNORECASE,
    )

    has_po_code = re.search(
        r"\bPO\s*10\b",
        question,
        re.IGNORECASE,
    )

    if not has_course_code and not has_po_code:
        for message in reversed(history):
            if message["role"] != "user":
                continue

            previous_question = message["content"]

            previous_has_course = re.search(
                r"\b[A-Z]{2,}\s+\d{3}[A-Z]?\b",
                previous_question,
                re.IGNORECASE,
            )

            previous_has_po = re.search(
                r"\bPO\s*10\b",
                previous_question,
                re.IGNORECASE,
            )

            if previous_has_course or previous_has_po:
                retrieval_question = (
                    previous_question + "\n" + question
                )
                break

    results = retrieve_context(retrieval_question)

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

    # Pass history to the language model.
    answer = generate_answer(
        question,
        context,
        history=history,
    )

    sources = [
        {
            "file": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in results
    ]

    return {
        "answer": answer,
        "sources": sources,
    }

