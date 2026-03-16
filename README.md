# Telecom Data Analytical Agent (Text-to-SQL)

A robust, multi-agent conversational AI system built with LangGraph and FastAPI. This agent translates natural language business questions into accurate SQL queries, executes them against a local SQLite database, and returns human-readable insights. It is specifically tailored for Telecom domain data.

## 🌟 Key Features

* **Multi-Agent Architecture:** An Orchestrator (`AnalyticalAgent`) handles conversation and mathematical reasoning, delegating complex data retrieval to a specialized sub-agent (`Text2SQL`).
* **Self-Correcting SQL Generation:** Features a built-in feedback loop. If a generated SQL query fails syntax validation or contains semantic logic errors, the agent automatically debugs and rewrites the query before executing it.
* **Vector-Grounded Retrieval (RAG):** Uses ChromaDB to fetch database schemas, domain definitions (evidence), exact categorical values, and few-shot examples to ensure the LLM writes perfectly aligned SQL.
* **Built-in Analytical Tools:** Natively handles mathematical requests like percentage calculations and period-over-period growth comparisons without needing to write complex SQL for them.
* **Persistent Memory:** Utilizes a PostgreSQL checkpointer to maintain conversational context across different chat threads.
* **Modern REST API:** Fully asynchronous FastAPI backend with endpoints for session management and agent invocation.

## 📋 Prerequisites

Before you begin, ensure you have the following installed and running:

* **Python 3.11+**
* **PostgreSQL Server:** Required for LangGraph's conversation checkpointing.
* **Ollama (Local):** Must be running locally serving the `qwen3-embedding:4b` model for ChromaDB embeddings.
* **Groq API Key:** Required for the LLM models (`openai/gpt-oss-120b` and `qwen/qwen3-32b`).

## 🛠️ Project Structure

```text
├── main.py                # FastAPI application and endpoint definitions
├── analytical_agent.py    # Main orchestrator agent and calculation tools
├── text2sql_agent.py      # Sub-agent for Vector DB retrieval and SQL generation/checking
├── models.py              # Centralized LLM, ChromaDB, and Database configurations
├── prompts.py             # System and user prompts for all agent nodes
├── states.py              # Pydantic models and TypedDicts for LangGraph states
├── cockpit.db             # Your local SQLite target database
├── chroma_db_store/       # Local ChromaDB vector storage directory
└── .env                   # Environment variables

```

## ⚙️ Installation & Setup

**1. Clone the repository and navigate to the project directory.**

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install requirements.txt
```

**3. Configure Environment Variables:**
Create a `.env` file in the root directory and add your credentials:

```env
GROQ_API_KEY="your_groq_api_key_here"
DB_URI="postgresql://username:password@localhost:5432/your_database_name"

```

**4. Start Local Services:**

* Ensure your PostgreSQL server is active.
* Start Ollama in your terminal: `ollama serve`

## 🚀 Running the Application

Start the FastAPI server using Uvicorn:

```bash
fastapi dev main.py --reload

```

The server will start on `http://127.0.0.1:8000`. The startup lifespan event will automatically connect to PostgreSQL and set up the required checkpointing tables.

## 🌐 API Endpoints

You can interact with the API or view the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### 1. Create a New Chat

* **Endpoint:** `POST /chats/new`
* **Description:** Generates a new unique `chat_id` (thread ID) for a fresh conversation.
* **Response:**
```json
{
  "chat_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "New chat session created successfully"
}

```



### 2. Invoke the Agent (Ask a Question)

* **Endpoint:** `POST /chats/{chat_id}/ask`
* **Description:** Sends a natural language question to the agent using the specific `chat_id` to maintain conversation history.
* **Payload:**
```json
{
  "message": "What is the total number of active B2C customers on the iDar offer?"
}

```


* **Response:**
```json
{
  "chat_id": "123e4567-e89b-12d3-a456-426614174000",
  "response": "There are currently 45,210 active B2C customers on the iDar offer."
}

```
