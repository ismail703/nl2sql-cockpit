import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
import requests

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
DB_URI = os.getenv("DB_URI")

DB_PATH = "cockpit.db"

# Initialize LLMs
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=api_key,
    temperature=0, 
)

qwen = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=api_key,
    temperature=0, 
)

llama = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=api_key,
    temperature=0, 
)

client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=60,
)

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "qwen3-embedding:4b", "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]