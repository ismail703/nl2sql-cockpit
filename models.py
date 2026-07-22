import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from psycopg_pool import ConnectionPool
from langchain_groq import ChatGroq
from psycopg.rows import dict_row


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
COCKPIT_DB_URI = os.getenv("COCKPIT_DB_URI")
DB_CP_URI = os.getenv("DB_CP_URI")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=60,
)

gpt = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0, 
)

qwen = ChatGroq(
    model="qwen/qwen3.6-27b",
    groq_api_key=GROQ_API_KEY,
    temperature=0, 
)

embed_model = OpenAIEmbeddings(
    base_url=OLLAMA_BASE_URL,
    api_key='ollama',
    model='qwen3-embedding:4b',
    check_embedding_ctx_length=False,
)

cockpit_db_pool = ConnectionPool(
    conninfo=COCKPIT_DB_URI,
    kwargs={"autocommit": True, "row_factory": dict_row},
    min_size=1,
    max_size=10,
)