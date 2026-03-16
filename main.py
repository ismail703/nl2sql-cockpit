from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uuid

from analytical_agent import AnalyticalAgent
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
import os

app = FastAPI(
    title="Analytical Agent API",
    description="API for interacting with the Analytical Agent",
    version="1.0.0"
)

class NewChatResponse(BaseModel):
    chat_id: str = Field(description="The unique identifier for the new chat thread.")
    message: str = Field(description="Status message confirming creation.")

class ChatRequest(BaseModel):
    message: str = Field(description="The natural language question from the user.")

class ChatResponse(BaseModel):
    chat_id: str = Field(description="The thread ID used for this conversation.")
    response: str = Field(description="The final natural language answer from the agent.")


load_dotenv()
DB_URI = os.getenv("DB_URI")

@app.post("/chats/new", response_model=NewChatResponse, tags=["Chat Management"])
async def create_new_chat():    
    return NewChatResponse(
        chat_id=str(uuid.uuid4()),
        message="New chat session created successfully"
    )

@app.post("/chats/{chat_id}/ask", response_model=ChatResponse, tags=["Agent Interaction"])
async def invoke_agent(chat_id: str, request: ChatRequest):
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()             
            agent = AnalyticalAgent(checkpointer)            
            result = agent.chat(user_input=request.message, thread_id=chat_id)
        
        return ChatResponse(
            chat_id=chat_id,
            response=result
        )
        
    except Exception as e:
        print(f"[Error] Agent invocation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Agent Error: {str(e)}")