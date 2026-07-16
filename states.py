import operator
from typing import Annotated, TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# ==========================================
# Supervisor Agent States
# ==========================================

class RouteDecision(BaseModel):
    route: Literal["feedback", "analytical", "unrelated"] = Field(
        description=(
            "Classify the user's latest message: "
            "'feedback' if it's a correction, note, remark, or comment about a past answer; "
            "'analytical' if it's a data/insight/SQL request; "
            "'unrelated' if it's a greeting, small talk, or anything not tied to data or feedback."
        )
    )

class FeedbackEvaluation(BaseModel):
    is_approved: bool = Field(
        description="True if the user approved the task without needing changes. False if they provided corrections, new instructions, or rejected it."
    )
    updated_task_description: str = Field(
        description="The final task description. MUST BE PLAIN TEXT ONLY. Do not use markdown, newlines (\\n), or bullet points. Write as a single continuous paragraph. If approved, return the original."
    )

class MemoryReconciliation(BaseModel):
    action: Literal["add", "update", "delete", "skip"] = Field(
        description=(
            "'add' if this is genuinely new knowledge with no close match; "
            "'update' if it refines/corrects an existing similar lesson (merge into it); "
            "'delete' if it explicitly invalidates/contradicts an existing lesson and nothing should replace it; "
            "'skip' if it's a duplicate or near-duplicate of an existing lesson with no new information."
        )
    )
    target_id: Optional[str] = Field(
        default=None,
        description="ID of the existing lesson to update or delete. Required for 'update' and 'delete', null otherwise."
    )
    final_lesson: Optional[str] = Field(
        default=None,
        description="The lesson text to store, for 'add' or 'update'. Null for 'delete' or 'skip'."
    )

class SupervisorState(MessagesState):
    task_description: str
    final_result: str
    sub_questions: List[str]
    data_results: Annotated[List[str], operator.add]
    analytical_request: str
    memory_context: Optional[str]
    correction_notes: Optional[str]
    route_decision: Optional[RouteDecision]

class Text2SQLRequests(BaseModel):
    sub_questions: List[str]
    data: List[str]

# ==========================================
# Text2SQL Agent States
# ==========================================

class VectorDBQueries(BaseModel):
    """Output model for generating targeted queries for each Vector DB"""
    schema_query: List[str] = Field(description="List of queries to fetch schema information (table/column)")
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
    db_results: Annotated[List[str], operator.add]   # Results of all vector db
    sql_candidate: str                               # Generated SQL Query 
    is_sql_modified: bool                            # Flag to trigger the feedback loop
    query_result: str                                # Stores the successful data
    syntax_retry: int                                # Safety limit
    semantic_retry: int
    formatted_result: str
    data_results: Annotated[List[str], operator.add]

# ==========================================
# Research Agent States
# ==========================================

class SearchTask(TypedDict):
    query: str
    category: str

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

