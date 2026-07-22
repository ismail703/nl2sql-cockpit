# Telecom Data Analytical Agent

This project is a FastAPI application built with LangGraph for telecom analytics. It turns natural-language business questions into SQL, executes them against the Cockpit PostgreSQL database, and returns a readable answer.

## What Changed

The project now uses PostgreSQL for Cockpit data instead of SQLite. The current workflow focuses on supervisor-driven SQL analysis and no longer includes the research agent or Tavily-based external enrichment.

## Architecture

The application is organized around a small, focused workflow:

- `SupervisorAgent` handles the incoming chat request, planning, human-in-the-loop review, and final reasoning.
- `Text2SQL` generates SQL, validates it, and executes it against the Cockpit PostgreSQL database.

Retrieval still relies on Qdrant. The scripts under `retrieve/` index schema, values, evidence, and examples from the JSON files in `context/`.

## PostgreSQL Setup

Run PostgreSQL locally in Docker before starting the application:

```bash
docker run -d --name cockpit-postgres -e POSTGRES_USER=username -e POSTGRES_PASSWORD=password -e POSTGRES_DB=cockpit_db -p 5432:5432 postgres:latest
```

This command creates a container named `cockpit-postgres` and exposes PostgreSQL on `localhost:5432`.

### Optional database initialization

After the database is running, create the Cockpit tables and seed the mock data by running the database setup script used by the project.

If you maintain your own initialization script, the expected database is:

- Host: `localhost`
- Port: `5432`
- Database: `cockpit_db`
- Username: `username`
- Password: `password`

## Environment Configuration

Create a `.env` file in the repository root and configure the services used by the app:

```env
COCKPIT_DB_URI=postgresql://root:1234@localhost:5432/cockpit_db
GROQ_API_KEY=your_groq_api_key_here

# Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Notes:

- `COCKPIT_DB_URI` is required for the PostgreSQL connection.
- `GROQ_API_KEY` is required for the LLM calls.
- `LANGFUSE_HOST` can point to Langfuse Cloud or a self-hosted instance.

## Retrieval Data Flow

The JSON files that feed the retrieval pipeline should be placed in `context/`:

- `context/db_schema.json`
- `context/db_values.json`
- `context/evidence.json`
- `context/question-example.json`

Keep these files in `context/` so the scripts in `retrieve/` can load them without changing paths.

## Qdrant Setup

Run Qdrant locally in Docker before populating the vector collections:

```bash
docker volume create qdrant_storage
docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage --restart unless-stopped qdrant/qdrant
```

The application connects to Qdrant on `localhost:6333`.

## Retrieve Folder

The `retrieve/` folder contains the indexing scripts that build the Qdrant collections used by the app:

- `store_db_schema.py` indexes the database schema from `context/db_schema.json`.
- `store_category_db.py` indexes distinct column values from `context/db_values.json`.
- `store_evidence.py` indexes domain evidence from `context/evidence.json`.
- `store_examples.py` indexes few-shot question and SQL examples from `context/question-example.json`.

Run these scripts directly as modules after Qdrant is running and the JSON files are in place.

Example:

```bash
python -m retrieve.store_category_db
python -m retrieve.store_db_schema
python -m retrieve.store_evidence
python -m retrieve.store_examples
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start PostgreSQL with the Docker command above.
4. Create the `.env` file with `COCKPIT_DB_URI` and the remaining secrets.
5. Start Qdrant with the Docker commands above.
6. Make sure the JSON files are present in `context/`.
7. Run the retrieval scripts in `retrieve/` to populate Qdrant.
8. Start the API with `fastapi dev main.py --reload`.

## API

The app exposes two main endpoints:

- `POST /chats/new` creates a new chat session and returns a `chat_id`.
- `POST /chats/{chat_id}/ask` sends a user message through the supervisor workflow.

Example:

```bash
curl -X POST http://127.0.0.1:8000/chats/new -H "Content-Type: application/json"
curl -X POST http://127.0.0.1:8000/chats/<chat_id>/ask -H "Content-Type: application/json" -d "{\"message\":\"Give me the MTD comparison for product *6\"}"
```

## Project Structure

```text
main.py                  # FastAPI app and request lifecycle
models.py                # LLM, embeddings, and Qdrant configuration
prompts.py               # Prompt templates used by the agents
states.py                # LangGraph state definitions
agents/
  supervisor_agent.py    # Main orchestrator and reasoning flow
  text2sql.py            # SQL generation and execution agent
retrieve/                # Qdrant indexing scripts for schema, values, evidence, and examples
context/                 # JSON source files used to populate Qdrant collections
assets/                  # Static assets and local storage
requirements.txt         # Python dependencies
README.md                # Project documentation
```

## Notes

- The supervisor workflow supports human review pauses before querying the database.
- The PostgreSQL checkpointer keeps conversation state aligned with the `chat_id` thread.
- Langfuse is the recommended place to inspect execution traces when debugging agent behavior.
- If a sub-query returns an empty result, verify that Qdrant is running, the retrieval scripts have been executed, and the JSON files in `context/` contain the expected data.
