import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq


def _get_llm(temperature: float = 0.1) -> BaseChatModel:
    """Return a configured LLM. Swap provider via LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    if provider == "groq":
        return ChatGroq(model=model, temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: groq")
