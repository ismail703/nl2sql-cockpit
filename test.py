from supervisor_agent import SupervisorAgent
from text2sql import Text2SQL
from langgraph.checkpoint.postgres import PostgresSaver
from models import DB_URI

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()                         
    supervisor = SupervisorAgent(checkpointer=checkpointer)
    text2sql_agent = Text2SQL()
    config = {"configurable": {"thread_id": "kl62ufbf-95c3-4a05-be5d-0b060a07c0f9"}}


    current_state = supervisor.agent.get_state(config)
    print("Current State Snapshot:")
    print(current_state)