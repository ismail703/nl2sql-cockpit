import uuid
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langfuse.langchain import CallbackHandler

from agents.supervisor_agent import SupervisorAgent

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("DB_URI environment variable is not set.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_pool = None
supervisor_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI to handle global resources safely."""
    global db_pool, supervisor_agent

    logger.info("Initializing database pool and Supervisor Agent...")
    db_pool = ConnectionPool(
        conninfo=DB_URI,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
        },
    )
    checkpointer = PostgresSaver(db_pool)
    checkpointer.setup()

    supervisor_agent = SupervisorAgent(checkpointer=checkpointer)

    yield

    logger.info("Closing database pool...")
    if db_pool:
        db_pool.close()

# FastAPI App
app = FastAPI(
    title="Analytical Agent API",
    description="API for interacting with the Analytical Supervisor Agent with HITL capabilities",
    version="2.0.0",
    lifespan=lifespan
)

app = FastAPI(
    title="Analytical Agent API",
    description="API for interacting with the Analytical Supervisor Agent with HITL capabilities",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React/Vite frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewChatResponse(BaseModel):
    chat_id: str
    message: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    chat_id: str
    response: str
    status: str = "completed"
    task_description: Optional[str] = None


def get_supervisor():
    """Dependency injection for the supervisor."""
    if not supervisor_agent:
        raise HTTPException(
            status_code=500, detail="Agent not initialized properly.")
    return supervisor_agent


@app.post("/chats/new", response_model=NewChatResponse, tags=["Chat"])
async def create_new_chat():
    chat_id = str(uuid.uuid4())
    logger.info(f"[NEW CHAT] {chat_id}")

    return NewChatResponse(
        chat_id=chat_id,
        message="New chat session created successfully"
    )


@app.post("/chats/{chat_id}/ask", response_model=ChatResponse, tags=["Agent"])
async def invoke_agent(
    chat_id: str,
    request: ChatRequest,
    supervisor: SupervisorAgent = Depends(get_supervisor)
):
    try:
        langfuse_handler = CallbackHandler()
        config = {
            "configurable": {"thread_id": chat_id},
            "callbacks": [langfuse_handler]
        }

        current_state = supervisor.agent.get_state(config)

        if current_state.next and "human_review" in current_state.next:
            logger.info(f"[RESUME] {chat_id} | Feedback: {request.message}")
            supervisor.agent.invoke(
                Command(resume=request.message), config=config)
        else:
            logger.info(f"[START] {chat_id} | Query: {request.message}")
            supervisor.agent.invoke(
                {"messages": [HumanMessage(content=request.message)]},
                config=config
            )

        new_state = supervisor.agent.get_state(config)

        if new_state.next and "human_review" in new_state.next:
            pending_task = new_state.tasks[0]
            interrupt_payload = pending_task.interrupts[0].value

            return ChatResponse(
                chat_id=chat_id,
                response=interrupt_payload["message"],
                status=interrupt_payload["status"],
                task_description=interrupt_payload.get("task_description")
            )

        final_message = new_state.values["messages"][-1].content

        return ChatResponse(
            chat_id=chat_id,
            response=final_message,
            status="completed"
        )

    except Exception as e:
        logger.error(f"[ERROR] {chat_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal Agent Error"
        )
