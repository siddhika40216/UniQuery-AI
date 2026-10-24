# UniQuery AI 🎓
### A Retrieval-Augmented Generation (RAG) Chatbot for University Information

UniQuery AI is an AI-powered university information assistant that answers questions using university course documents. It combines semantic search, vector-based retrieval, and a Large Language Model (LLM) to provide relevant answers grounded in the available university documents.

The system allows users to ask questions about courses, course codes, learning outcomes, and other available academic information through a simple web-based chat interface. It also supports follow-up questions using conversation history.

## ✨ Features

- **Document-Based Question Answering:** Retrieves relevant information from university course documents.
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with LLM-based answer generation.
- **Semantic Search:** Uses sentence embeddings and FAISS to retrieve relevant document chunks.
- **Course Code Retrieval:** Supports targeted searches for course codes such as `GEOG 203L`.
- **Programme Outcome Retrieval:** Includes targeted retrieval for PO10 definitions available in the extracted document.
- **Conversational Follow-Ups:** Uses recent conversation history to interpret follow-up questions.
- **Source Display:** Shows retrieved source information alongside generated answers.
- **Local LLM Inference:** Uses Ollama with Llama 3.2, allowing local answer generation.
- **Web-Based Chat Interface:** Provides a simple interface for interacting with the assistant.
- **New Chat:** Allows users to reset the current conversation.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend API | FastAPI |
| Language Model | Llama 3.2 (3B) |
| Local LLM Runtime | Ollama |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Data Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| HTTP Requests | Requests |
| Document Processing | PyMuPDF |

## 🧠 How It Works

UniQuery AI follows a Retrieval-Augmented Generation (RAG) pipeline.

1. **Document Ingestion:** Extracts text from the university course PDF.
2. **Text Processing:** Organizes the extracted content into course-level chunks.
3. **Embedding Generation:** Converts text chunks into numerical vector representations using Sentence Transformers.
4. **Vector Indexing:** Stores embeddings in a FAISS index for efficient similarity search.
5. **Query Processing:** Accepts a question from the user through the chat interface.
6. **Context Retrieval:** Retrieves relevant document chunks using semantic search and targeted retrieval rules.
7. **Answer Generation:** Passes the question, retrieved context, and recent conversation history to Llama 3.2 through Ollama.
8. **Response Display:** Returns the generated answer and available source information through the FastAPI backend.

```text
University Course PDF
        |
        v
   Text Extraction
        |
        v
  Course-Level Chunking
        |
        v
 Sentence Embeddings
        |
        v
     FAISS Index
        |
        v
    User Question
        |
        v
  Relevant Retrieval
        |
        v
 Retrieved Context + Chat History
        |
        v
 Ollama (Llama 3.2 3B)
        |
        v
 Answer + Source Information
        |
        v
    Web Chat Interface
```

## 📁 Project Structure

```text
UniQuery-AI/
├── src/
│   ├── ingest.py       # Extracts text from source documents
│   ├── chunking.py     # Creates structured course-level chunks
│   ├── retrieve.py     # Retrieves relevant document content
│   ├── generate.py     # Generates answers using Ollama
│   ├── chat.py         # Chat-related functionality
│   └── api.py          # FastAPI backend and API endpoints
│
├── frontend/
│   └── index.html      # Chat interface
│
├── data/
│   ├── documents/      # Source documents
│   ├── extracted/      # Extracted text
│   └── chunks/         # Processed chunks and vector index
│
├── .gitignore
└── README.md
```

*Note: Generated data files, model files, and local environment files may be excluded from Git using `.gitignore`.*

## ⚙️ Installation and Setup

### Prerequisites

Install or prepare the following:

- Python 3.10 or a compatible version for the project's dependencies
- Git
- Ollama
- Sufficient memory and disk space for the embedding model and local LLM

### 1. Clone the Repository

```bash
git clone https://github.com/siddhika40216/UniQuery-AI.git
cd UniQuery-AI
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution policy for your environment or activate the environment through Command Prompt instead.

### 3. Install Dependencies

Install the Python packages required by the project:

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu requests pymupdf
```

### 4. Install and Start Ollama

Install Ollama from its official website:

https://ollama.com/

Download the model used by the project:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running locally before starting the chatbot.

### 5. Prepare the University Documents

Place the source university PDF in the appropriate location under `data/documents/`.

Run the document-processing scripts from the project root in the order required by the pipeline:

```bash
python src/ingest.py
python src/chunking.py
```

These scripts prepare the extracted text, chunks, embeddings, and vector index required for retrieval.

**Important:** The exact input filename and output paths must match the configuration in the scripts. If processed data files are excluded from Git, regenerate them locally before using the application.

### 6. Start the FastAPI Server

From the project root, run:

```bash
uvicorn src.api:app --reload
```

Open the application in your browser:

http://127.0.0.1:8000

The FastAPI backend serves the chat interface and handles questions through the `/chat` endpoint.

## 💬 Usage

1. Open UniQuery AI in your browser.
2. Enter a question about the available university course documents.
3. Review the generated answer and its displayed sources.
4. Ask follow-up questions, for example:
   - `Tell me about GEOG 203L.`
   - `What are its learning outcomes?`
   - `Tell me more about it.`
5. Click **New Chat** to start a fresh conversation.

The quality of an answer depends on the information available in the source documents and the relevance of the retrieved context.

## 🔌 API Overview

### `POST /chat`

Accepts a user question and optional conversation history.

Example request:

```json
{
  "question": "Tell me about GEOG 203L.",
  "history": []
}
```

Example response structure:

```json
{
  "answer": "Generated answer based on the retrieved university document context.",
  "sources": []
}
```

The `sources` array contains source information returned by the backend. The exact fields and contents depend on the retrieval implementation and available documents.

## 🔒 Design Considerations

- Answers are intended to be grounded in the retrieved university document context.
- The local Ollama model generates responses without requiring a hosted LLM API for inference.
- Targeted retrieval rules improve handling of selected course-code and programme-outcome queries.
- Generated answers may still be incomplete or inaccurate if the source documents lack relevant information or retrieval misses the required context.

## 🚧 Limitations

- The current knowledge base depends on the university documents processed by the pipeline.
- The system does not automatically know information that is absent from its available documents.
- Retrieval quality depends on document structure, chunking, embeddings, and query wording.
- Local inference performance depends on the machine's available memory and processing resources.
- Conversation history is maintained by the current chat interface and request flow; persistent, cross-session conversation storage is not implemented.

## 🔮 Future Improvements

- Support multiple university documents and document formats.
- Improve retrieval accuracy with hybrid search and reranking.
- Add document upload and re-indexing through the interface.
- Improve source citations with more detailed page-level references.
- Add automated retrieval and answer evaluation.
- Improve error handling and loading states.
- Add persistent conversation storage and chat history management.
- Add authentication and user-specific sessions if required.

## 🎯 Project Objective

The goal of UniQuery AI is to make university academic information easier to access through a conversational interface, reducing the effort required to search lengthy course documents manually.

## 👩‍💻 Author

**Siddhika Srivastava**

B.Tech Computer Science and Engineering  
Banasthali Vidyapith.

---

*Built as an AI-powered university information assistant using Retrieval-Augmented Generation, semantic search, and local LLM inference.*
