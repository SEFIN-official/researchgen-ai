import os
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv())

from langchain_google_genai import ChatGoogleGenerativeAI
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


#For openAI
"""from langchain.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )"""