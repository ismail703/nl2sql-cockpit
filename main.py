from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import os
from dotenv import load_dotenv

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from supervisor_agent import SupervisorAgent 

load_dotenv()
DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError("DB_URI environment variable is not set. Please check your .env file.")


app = FastAPI(
    title="Analytical Agent API",
    description="API for interacting with the Analytical Supervisor Agent with HITL capabilities",
    version="1.0.0"
)

class NewChatResponse(BaseModel):
    chat_id: str = Field(description="The unique identifier for the new chat thread.")
    message: str = Field(description="Status message confirming creation.")

class ChatRequest(BaseModel):
    message: str = Field(description="The natural language question OR feedback to a proposed task.")

class ChatResponse(BaseModel):
    chat_id: str = Field(description="The thread ID used for this conversation.")
    response: str = Field(description="The final answer OR the prompt asking for task approval.")
    status: str = Field(default="completed", description="'completed' or 'awaiting_approval'")
    task_description: Optional[str] = Field(default=None, description="The generated task description waiting for review.")


@app.post("/chats/new", response_model=NewChatResponse, tags=["Chat Management"])
async def create_new_chat():    
    """Creates a new conversation thread ID."""
    return NewChatResponse(
        chat_id=str(uuid.uuid4()),
        message="New chat session created successfully"
    )

@app.post("/chats/{chat_id}/ask", response_model=ChatResponse, tags=["Agent Interaction"])
async def invoke_agent(chat_id: str, request: ChatRequest):
    """
    Handles both new questions and feedback for paused (interrupted) tasks.
    It automatically checks the graph state to determine how to route the request.
    """
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()             
            
            supervisor = SupervisorAgent(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": chat_id}}
            
            current_state = supervisor.agent.get_state(config)
            
            if current_state.next and "human_review" in current_state.next:
                print(f"[API] Resuming graph {chat_id} with feedback: '{request.message}'")
                supervisor.agent.invoke(Command(resume=request.message), config=config)
            else:                
                print(f"[API] Starting new run for {chat_id} with query: '{request.message}'")
                supervisor.agent.invoke({"messages": [HumanMessage(content=request.message)]}, config=config)
            
            new_state = supervisor.agent.get_state(config)
            
            if new_state.next and "human_review" in new_state.next:                
                pending_task = new_state.tasks[0]
                interrupt_payload = pending_task.interrupts[0].value
                
                return ChatResponse(
                    chat_id=chat_id,
                    response=interrupt_payload["message"],
                    status=interrupt_payload["status"],
                    task_description=interrupt_payload["task_description"]
                )
            else:            
                final_message = new_state.values["messages"][-1].content
                return ChatResponse(
                    chat_id=chat_id,
                    response=final_message,
                    status="completed",
                    task_description=None
                )
                
    except Exception as e:
        print(f"[Error] Agent invocation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Agent Error: {str(e)}")