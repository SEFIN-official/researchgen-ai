import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain_google_genai import ChatGoogleGenerativeAI


def _api_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def get_llm(temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=_api_key(),
    )


def get_judge_llm():
    """Lower temperature for classify / critic structured decisions."""
    return get_llm(temperature=0)
