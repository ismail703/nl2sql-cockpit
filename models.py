import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
DB_URI = os.getenv("DB_URI")

DB_PATH = "cockpit.db"
CHROMA_PATH = "./chroma_db_store"
MODEL_NAME = "granite3.2:2b"
QWEN_MODEL = "qwen3:1.7b"

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

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path=CHROMA_PATH)
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:4b"
)

coll_schema = client.get_collection("telco_db_schema", embedding_function=ollama_ef)
coll_evidence = client.get_collection("telco_domain_evidence", embedding_function=ollama_ef)
coll_values = client.get_collection("telco_distinct_values", embedding_function=ollama_ef)
coll_examples = client.get_collection("sql_few_shot_examples", embedding_function=ollama_ef)