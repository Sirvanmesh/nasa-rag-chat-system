"""OpenAI chat client for the NASA mission RAG assistant."""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Number of previous chat messages (user + assistant) sent with each request.
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """You are a NASA mission expert and historian. You answer questions \
about the Apollo 11, Apollo 13, and Challenger (STS-51L) missions.

Rules:
1. Answer only with facts from the retrieved context in the user's message. \
Do not use outside knowledge to add facts.
2. Cite your sources inline with their numbers, for example [Source 1] or [Source 2, Source 3].
3. If the context does not contain enough information to answer, say so clearly, \
for example: "The retrieved documents do not contain enough information to answer this." \
Then share only what the context does support.
4. Be accurate, clear, and concise. Keep a respectful tone, especially when \
discussing the Challenger accident."""


def get_openai_client(openai_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client using the given key or the OPENAI_API_KEY env variable.

    OPENAI_BASE_URL is used when set, so Vocareum keys work without code changes.
    """
    api_key = openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY in your .env file.")
    return OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)


def build_messages(user_message: str, context: str,
                   conversation_history: List[Dict]) -> List[Dict[str, str]]:
    """Build the message list: system prompt, recent history, then context + question.

    The retrieved context is attached only to the current question. Older turns are
    sent as plain role/content pairs, so old contexts are never sent again.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for message in conversation_history[-MAX_HISTORY_MESSAGES:]:
        if message.get("role") in ("user", "assistant") and message.get("content"):
            messages.append({"role": message["role"], "content": message["content"]})

    if context.strip():
        user_content = (
            f"Retrieved context:\n{context}\n\n"
            f"Question: {user_message}\n\n"
            "Answer using only the context above and cite the sources."
        )
    else:
        user_content = (
            "Retrieved context: (no relevant documents were found)\n\n"
            f"Question: {user_message}\n\n"
            "Tell the user that the documents do not contain enough information to answer."
        )
    messages.append({"role": "user", "content": user_content})
    return messages


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate a grounded answer from OpenAI using the retrieved context and chat history."""
    if not user_message or not user_message.strip():
        raise ValueError("User message is empty.")

    client = get_openai_client(openai_key)
    messages = build_messages(user_message, context or "", conversation_history or [])

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()
