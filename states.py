import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# ==========================================
# Analytical Agent States
# ==========================================

class SubQuestionPlan(BaseModel):
    """Output model for the Planner Node"""
    sub_questions: List[str] = Field(
        description="List of simple, individual natural language questions to fetch necessary data."
    )

class AnalyticalState(MessagesState):
    sub_questions: List[str]
    data_results: Annotated[List[dict], operator.add]

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