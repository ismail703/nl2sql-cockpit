# Telecom Data Analytical Agent (Text-to-SQL)

A robust, multi-agent conversational AI system built with LangGraph and FastAPI. This agent translates natural language business questions into accurate SQL queries, executes them against a local SQLite database, and returns human-readable insights. It is specifically tailored for Telecom domain data.

## 🌟 Key Features

* **Multi-Agent Architecture:** An Orchestrator (`SupervisorAgent`) handles conversation and mathematical reasoning, delegating complex data retrieval to a specialized sub-agent (`Text2SQL`).
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
├── supervisor_agent.py    # Main orchestrator (HITL + orchestration) agent and calculation tools 
generation/checking
├── models.py              # Centralized LLM, ChromaDB, and Database configurations
├── prompts.py             # System and user prompts for all agent nodes
├── states.py              # Pydantic models and TypedDicts for LangGraph states
├── cockpit.db             # Your local SQLite target database
├── chroma_db_store/       # Local ChromaDB vector storage directory
├── text2sql.py            # Sub-agent for Vector DB retrieval and SQL generation/checking
└── .env                   # Environment variables
```
**1. Clone the repository and navigate to the project directory.**

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install requirements.txt
## 🔁 Updated Architecture (Supervisor + Analytical Fusion)

This repository now uses a Supervisor-centric workflow that merges responsibilities from the previous `AnalyticalAgent`. The high-level flow:

- **SupervisorAgent (orchestrator)**: Receives user messages via the FastAPI endpoint, generates a clean task description, optionally pauses for human review (HITL), plans sub-questions, and coordinates sub-agents.
- **Text2SQL sub-agent**: For each planned sub-question, the Supervisor dispatches to the `Text2SQL` sub-agent which:
  - Generates vector DB queries (schema, evidence, values, examples) using ChromaDB
  - Builds a contextual prompt and generates SQL via the LLM
  - Executes SQL against the local SQLite DB and auto-fixes syntax/semantic issues using a debug-and-rewrite loop
  - Formats the raw DB results into a human-readable answer
- **Final reasoning step**: Supervisor gathers all sub-agent outputs and runs a final reasoning pass (with lightweight calculation tools available to the LLM) to produce the final user response.

This design preserves modularity (specialized Text2SQL agent) while providing a single orchestration point (Supervisor) for HITL, planning, and final assembly.

## ✅ Components

- **FastAPI (`main.py`)**: Entrypoint; manages lifecycle resources (DB pool, checkpointer) and exposes `/chats/new` and `/chats/{chat_id}/ask` endpoints.
- **SupervisorAgent (`supervisor_agent.py`)**: Orchestrator that: task-generation → human-review interrupt → planning → sub-query dispatch → result aggregation → final reasoning.
- **Text2SQL (`text2sql.py`)**: Vector-grounded SQL generator and executor. Steps: vector retrieval → SQL generation → syntax check & auto-fix → semantic review → format result.
- **LLMs & Embeddings (`models.py`)**: LLM clients and ChromaDB collections used for retrieval and generation.
- **Persistence**: PostgreSQL checkpointer (LangGraph) for conversation state; SQLite for query execution; ChromaDB for RAG retrieval.
- **Tools**: Small analytic utilities (e.g., `calculate_percentage`, `compare_periods`) made available to the LLM during final reasoning.

## 🧩 Supervisor Schema

The supervisor writes its schema snapshot to [assets/supervisor_agent_schema.json](assets/supervisor_agent_schema.json) when it initializes. The snapshot captures the `SupervisorState` fields, the structured output models used by the planner and review nodes, the available calculation tools, and the workflow node list.

```mermaid
flowchart TD
  A[SupervisorAgent] --> B[SupervisorState]
  A --> C[FeedbackEvaluation]
  A --> D[SubQuestionPlan]
  A --> E[Text2SQLRequests]
  A --> F[Tool Set\ncalculate_percentage\ncompare_periods]
  A --> G[Workflow Nodes]
  A --> H[assets/supervisor_agent_schema.json]
  H --> I[README documentation]
```

## ⚠️ Known Setup Caveats

- The Postgres-based checkpointer runs migrations at startup which may include `CREATE INDEX CONCURRENTLY`. That SQL cannot run inside a transaction block and may fail under some pool/driver configurations. If you see errors about `CREATE INDEX CONCURRENTLY cannot run inside a transaction block`:
  - Run migrations once manually with an autocommit connection, or
  - Start the app with migration skipped (the code includes a startup guard to skip the failing migration and continue; consult `main.py`).

## 🚀 Quick Start

1. Create and activate a Python environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Provide environment variables in `.env`:

```env
GROQ_API_KEY="your_groq_api_key"
DB_URI="postgresql://user:pass@localhost:5432/your_db"
```

3. Ensure external services are running: PostgreSQL, Ollama (if used for local embeddings), any embedding endpoint expected by `models.py`.

4. Start the API:

```bash
fastapi dev main.py --reload
```

5. Create a chat and ask a question (example using `curl`):

```bash
curl -X POST http://127.0.0.1:8000/chats/new -H "Content-Type: application/json"
curl -X POST http://127.0.0.1:8000/chats/<chat_id>/ask -H "Content-Type: application/json" -d '{"message":"Give me the MTD comparison for product *6"}'
```

## 🧪 Testing & Debugging Tips

- Enable more verbose logging in `main.py` to trace agent lifecycles and message flows.
- When a sub-query returns an empty result (`"No result generated"`), inspect the vector DB retrieval steps in `text2sql.py` to ensure ChromaDB collections are populated and embedding service is reachable.
- For LLM-toolcall validation errors (e.g., Groq tool validation), the supervisor includes fallbacks that avoid structured tool-calling in key nodes — see `supervisor_agent.py`'s conversation-history and reasoning nodes.

## Next Improvements

- Add unit tests for each StateGraph node.
- Add a controlled migration runner that executes index statements with autocommit.
- Improve retry/backoff for external services (ChromaDB, Embedding API, LLM endpoints).

---

For implementation details, see the core modules: [main.py](main.py), [supervisor_agent.py](supervisor_agent.py), [text2sql.py](text2sql.py), and [models.py](models.py).
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
