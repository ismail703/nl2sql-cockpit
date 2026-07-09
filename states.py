import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# ==========================================
# Supervisor Agent States
# ==========================================

class FeedbackEvaluation(BaseModel):
    is_approved: bool = Field(
        description="True if the user approved the task without needing changes. False if they provided corrections, new instructions, or rejected it."
    )
    updated_task_description: str = Field(
        description="The final task description. If approved, return the original. If not approved, rewrite the original task description to incorporate the user's feedback."
    )

class SupervisorState(MessagesState):
    task_description: str
    final_result: str
    sub_questions: List[str]
    data_results: Annotated[List[str], operator.add]
    analytical_request: str

class Text2SQLRequests(BaseModel):
    sub_questions: List[str]
    data: List[str]

class SubQuestionPlan(BaseModel):
    """Output model for the Planner Node"""
    sub_questions: List[str] = Field(
        description="List of simple, individual natural language questions to fetch necessary data."
    )


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
    formatted_result: str
    data_results: Annotated[List[str], operator.add]

class SearchTask(TypedDict):
    """Payload sent to each parallel search_node invocation via Send."""
    query: str
    category: str  # "explanation" | "competitor"


class SearchFinding(TypedDict):
    category: str
    query: str
    result: str


class QueryPlan(BaseModel):
    explanation_queries: List[str] = Field(
        description="Queries to research general/behavioral explanations for the finding."
    )
    competitor_queries: List[str] = Field(
        description="Queries to check for related Orange Maroc / Maroc Telecom offers."
    )


class ResearchAgentState(TypedDict):
    analytical_finding: str
    queries: List[SearchTask]
    search_results: Annotated[List[SearchFinding], operator.add]
    report: str

