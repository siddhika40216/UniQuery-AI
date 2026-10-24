
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_answer(question, context, history=None):
    if history is None:
        history = []

    conversation_history = ""

    for message in history[-6:]:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = message.role
            content = message.content

        conversation_history += (
            f"{role.capitalize()}: {content}\n"
        )

    prompt = f"""
You are UniQuery AI, a helpful university information assistant.

Use the provided university document context to answer questions.

Rules:
- Answer using only relevant information from the university document context.
- Do not invent university facts.
- Keep course codes and technical terms unchanged.
- Give a clear, simple answer.
- Use conversation history to understand follow-up questions
  such as "What are its learning outcomes?", "tell me more",
  and references like "it", "its", or "that course".
- If the user asks "What are your views about [course]?",
  interpret it as a request for a course overview based on the context.
- Do not claim to have personal opinions.
- If the context contains relevant information, answer using it.
- If the context does not contain the requested information,
  say: "I could not find this information in the available university documents."

Conversation history:
{conversation_history}

Current question:
{question}

University document context:
{context}

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
