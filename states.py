import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class SubQuestionPlan(BaseModel):
    """Output model for the Planner Node"""
    sub_questions: List[str] = Field(
        description="List of simple, individual natural language questions to fetch necessary data."
    )

class FeedbackEvaluation(BaseModel):
    is_approved: bool = Field(
        description="True if the user approved the task without needing changes. False if they provided corrections, new instructions, or rejected it."
    )
    updated_task_description: str = Field(
        description="The final task description. If approved, return the original. If not approved, rewrite the original task description to incorporate the user's feedback."
    )


class AnalyticalState(TypedDict):
    original_question: str
    sub_questions: List[str]
    data_results: Annotated[List[dict], operator.add]

class SupervisorState(MessagesState):
    task_description: str
    final_result: str
    data_results: List[dict]

# ==========================================
# Text2SQL Agent States
# ==========================================

class VectorDBQueries(BaseModel):
    """Output model for generating targeted queries for each Vector DB"""
    schema_query: List[str] = Field(description="List of queries to find similar SQL patterns")
    knowledge_query: List[str] = Field(description="List of queries for domain rules")
    value_query: List[str] = Field(description="List of queries for specific data values")
    example_query: List[str] = Field(description="List of queries to find similar SQL patterns")

class SemanticCheckResult(BaseModel): 
    reasoning: str = Field(description="Explanation of why the SQL is correct or incorrect based on the user question.") 
    is_semantically_correct: bool = Field(description="True if the SQL perfectly matches the user intent. False if logic needs fixing.") 
    corrected_sql: str = Field(description="The fixed SQL query if incorrect. If correct, return the original SQL.") 

class AgentState(TypedDict):
    question: str                                    # User's question    
    vect_queries: dict                               # Queries to fetch data from vector db each query with it's key
    db_results: Annotated[List[dict], operator.add]  # Results of all vector db
    sql_candidate: str                               # Generated SQL Query 
    is_sql_modified: bool                            # Flag to trigger the feedback loop
    query_result: str                                # Stores the successful data
    retry_count: int                                 # Safety limit
    final_output: str  
    data_results: Annotated[List[dict], operator.add]