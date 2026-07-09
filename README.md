# Telecom Data Analytical Agent

This project is a multi-agent FastAPI application built with LangGraph. It turns natural-language business questions into SQL, executes them against the local database, and returns a readable answer for telecom analytics workflows.

## Overview

The application is organized around a supervisor-first workflow:

- `SupervisorAgent` handles the incoming chat request, planning, human-in-the-loop review, final reasoning, and orchestration.
- `Text2SQL` generates and validates SQL for structured database questions.
- `ResearchAgent` takes the final analytical finding and expands it with external research using Tavily search, then synthesizes a short research report.

The response flow keeps SQL work focused while allowing the supervisor to combine database results, calculations, and research into a single answer.

## Langfuse Tracing

Langfuse is used to trace each chat turn and track the agent workflow across requests. In `main.py`, the FastAPI request handler creates a Langfuse callback handler so the supervisor and its LangGraph execution can be observed in Langfuse.

Recommended usage:

- Set Langfuse credentials in `.env` before starting the app.
- Keep the Langfuse callback attached to the request lifecycle so each `chat_id` stays traceable as a session.
- Use descriptive trace and session names, and avoid logging sensitive prompt or database values.

If Langfuse is configured correctly, you can inspect the full chat turn, nested agent steps, and any human-review pauses from the Langfuse UI.

## Project Structure

```text
main.py                  # FastAPI app and request lifecycle
models.py                # LLM, embeddings, and database configuration
prompts.py               # Prompt templates used by the agents
states.py                # LangGraph state definitions
agents/
  supervisor_agent.py    # Main orchestrator and reasoning flow
  text2sql.py            # SQL generation and execution agent
  research_agent.py      # External research and report synthesis agent
assets/                  # Static assets and local storage
chroma_db_store/         # Local ChromaDB vector store
requirements.txt         # Python dependencies
README.md                # Project documentation
```

## Research Agent

The `ResearchAgent` is a follow-up agent used after the supervisor produces an analytical finding. Its role is to:

1. Generate targeted research queries from the finding.
2. Search the web with Tavily for explanatory and competitor context.
3. Merge the retrieved evidence into a short report.

This is useful when the database answer needs additional market or product context before presenting the final response.

## Environment Configuration

Create a `.env` file in the repository root and add the required credentials:

```env
DB_URI=postgresql://username:password@localhost:5432/your_database_name
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

```

Notes:

- `DB_URI` is required for the PostgreSQL checkpointer.
- `GROQ_API_KEY` is required for the LLM calls.
- `TAVILY_API_KEY` is required by `ResearchAgent`.
- `LANGFUSE_HOST` can point to Langfuse Cloud or a self-hosted instance.

## Setup

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start PostgreSQL and any other local services used by your setup.
4. Start the API with `fastapi dev main.py --reload`.

## API

The app exposes two main endpoints:

- `POST /chats/new` creates a new chat session and returns a `chat_id`.
- `POST /chats/{chat_id}/ask` sends a user message through the supervisor workflow.

Example:

```bash
curl -X POST http://127.0.0.1:8000/chats/new -H "Content-Type: application/json"
curl -X POST http://127.0.0.1:8000/chats/<chat_id>/ask -H "Content-Type: application/json" -d "{\"message\":\"Give me the MTD comparison for product *6\"}"
```

## Notes

- The supervisor workflow supports human review pauses before querying the database.
- The PostgreSQL checkpointer keeps conversation state aligned with the `chat_id` thread.
- Langfuse is the recommended place to inspect execution traces when debugging agent behavior.
- When a sub-query returns an empty result (`"No result generated"`), inspect the vector DB retrieval steps in `text2sql.py` to ensure ChromaDB collections are populated and embedding service is reachable.
