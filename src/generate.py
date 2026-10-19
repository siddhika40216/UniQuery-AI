
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_answer(question, context):
    prompt = f"""
You are UniQuery AI, a helpful university information assistant.

Answer the user's question using ONLY the university document context below.

Rules:
- Do not invent university facts.
- If the answer is not present in the context, say:
  "I could not find this information in the available university documents."
- Give a clear, simple answer.
- Keep important course codes and technical terms unchanged.

University document context:
{context}

User question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    question = "Where is Banasthali Vidyapith located?"

    context = """
    Banasthali Vidyapith is located in Rajasthan, India.
    """

    print(generate_answer(question, context))
