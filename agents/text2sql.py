import sqlite3
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START 

from states import AgentState, VectorDBQueries, SemanticCheckResult
from prompts import (
    TEXT2SQL_DECOMPOSITION_SYSTEM_PROMPT,
    get_text2sql_generation_prompt,
    get_text2sql_debugger_system_prompt,
    get_text2sql_semantic_system_prompt,
    get_text2sql_semantic_user_prompt,
    TEXT2SQL_FORMAT_SYSTEM_PROMPT,
    get_text2sql_format_user_prompt
)
from models import gpt, qwen, client, embed_model, cockpit_db_pool


class Text2SQL:
    def __init__(self):
        self.agent = self.create_agent()
        self.cockpit_db_pool  = cockpit_db_pool
        self.client = client
      
    
    def generate_vect_db_query(self, state: AgentState):
        structured_llm = gpt.with_structured_output(VectorDBQueries)
        
        queries = structured_llm.invoke([
            SystemMessage(content=TEXT2SQL_DECOMPOSITION_SYSTEM_PROMPT),
            HumanMessage(content=state['question'])
        ])

        return {
            "vect_queries": {
                "schema": queries.schema_query,
                "evidence": queries.knowledge_query,
                "value": queries.value_query,
                "example": queries.example_query
            }
        }

    def retrieve_schema(self, state: AgentState):
        queries = state['vect_queries']

        res_schema = []
        for q_text in queries['schema']:
            query_vector = embed_model.embed_query(q_text)
            
            response = self.client.query_points(
                collection_name="telco_db_schema",
                query=query_vector,
                limit=1
            )
            
            if response.points:
                doc_content = response.points[0].payload.get("document", "")
                if doc_content.strip():
                    res_schema.append(doc_content)
        
        final_schema = "\n---\n".join(list(set(res_schema)))
        return {"db_results": [final_schema]}
   
    def retrieve_examples(self, state: AgentState):
        queries = state['vect_queries']
        search_term = queries.get("example", state['question'])
        if isinstance(search_term, list): search_term = search_term[0]

        query_vector = embed_model.embed_query(search_term)
        response = self.client.query_points(
            collection_name="sql_few_shot_examples",
            query=query_vector,
            limit=2
        )
        
        examples_list = []
        for point in response.points:
            doc_text = point.payload.get('document', 'No question found')
            sql_code = point.payload.get('query', 'No SQL found')
            examples_list.append(f"Question: {doc_text}\nSQL: {sql_code}")

        examples_txt = "\n---\n".join(examples_list) if examples_list else "No examples found."
        return {"db_results": [examples_txt]}
   
    def retrieve_evidence(self, state: AgentState):
        queries = state['vect_queries']
        
        search_terms = queries['evidence']
        unique_docs = set()
        
        for term in search_terms:
            query_vector = embed_model.embed_query(term)

            response = self.client.query_points(
                collection_name="telco_domain_evidence",
                query=query_vector,
                limit=3
            )
            
            for point in response.points:
                if point.payload and "document" in point.payload:
                    unique_docs.add(point.payload["document"])

        evidence_txt = "\n\n".join(list(unique_docs))
        return {"db_results": [evidence_txt]}

    def retrieve_values(self, state: AgentState):
        queries = state['vect_queries']
        
        search_terms = queries["value"]
        unique_value_mappings = set()
        
        for term in search_terms:
            query_vector = embed_model.embed_query(term)

            response = self.client.query_points(
                collection_name="telco_distinct_values",
                query=query_vector,
                limit=6
            )

            for point in response.points:
                meta = point.payload
                if meta:
                    val = meta.get('value', 'Unknown')
                    col = meta.get('column_name', 'Unknown')
                    tbl = meta.get('table_name', 'Unknown')
                    unique_value_mappings.add(f"Found Value: '{val}' in Table: {tbl}, Column: {col}")

        values_txt = "\n".join(list(unique_value_mappings))
        if not values_txt:
            values_txt = "No specific categorical value matches found."

        return {"db_results": [values_txt]}

    def generate_sql(self, state: AgentState):
        full_context = "\n\n".join(state['db_results'])
        
        prompt = get_text2sql_generation_prompt(full_context, state['question'])
        response = gpt.invoke([prompt])
        cleaned_sql = response.content.replace("```sql", "").replace("```", "").strip()

        return {"sql_candidate": cleaned_sql}

    def syntax_checker(self, state: AgentState):
        current_sql = state["sql_candidate"]
        retries = state.get("syntax_retry", 0)

        try:
            with self.cockpit_db_pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(current_sql)
                    result_data = cursor.fetchall()

            return {
                "query_result": str(result_data),
                "is_sql_modified": False,
                "syntax_retry": 0
            }

        except Exception as e:
            error_msg = str(e)

            if retries >= 3:
                return {
                    "is_sql_modified": False,
                    "query_result": f"Error: Failed after 3 attempts. Last error: {error_msg}",
                    "syntax_retry": 0,
                }

            full_context = "\n\n".join(state['db_results'])
            system_prompt = get_text2sql_debugger_system_prompt(full_context)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Original Query: {current_sql}\nPostgreSQL Error: {error_msg}")
            ]

            response = gpt.invoke(messages)
            fixed_sql = response.content.replace("```sql", "").replace("```", "").strip()

            return {
                "sql_candidate": fixed_sql,
                "is_sql_modified": True,
                "syntax_retry": retries + 1
            }

    def should_continue_syntax(self, state: AgentState) -> Literal["syntax_checker", "semantic_checker"]:
        return "syntax_checker" if state.get("is_sql_modified", False) else "semantic_checker"    

    def semantic_checker(self, state: AgentState):
        current_sql = state["sql_candidate"]
        original_question = state["question"]
        semantic_retry = state.get("semantic_retry", 0)
        structured_llm = qwen.with_structured_output(SemanticCheckResult)
        full_context = "\n\n".join(state['db_results'])
    
        system_prompt = get_text2sql_semantic_system_prompt(full_context)
        user_prompt = get_text2sql_semantic_user_prompt(original_question, current_sql)
    
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        is_correct = str(result.is_semantically_correct).strip().lower() == "true"

        if is_correct:
            return {"is_sql_modified": False}
        else:      
            if semantic_retry >= 1:
                return {
                    "is_sql_modified": False,
                    "semantic_retry": 0,
                }

            return {
                "sql_candidate": result.corrected_sql,
                "is_sql_modified": True,
                "syntax_retry": 0,
                "semantic_retry": semantic_retry + 1
            }

    def check_semantic_modification(self, state: AgentState) -> Literal["syntax_checker", "format_result"]:
        return "syntax_checker" if state.get("is_sql_modified", False) else "format_result"

    def format_result(self, state: AgentState):
        if "Error:" in state.get("query_result", ""):
            answer = f"I'm sorry, I encountered an issue: {state['query_result']}"
        else:
            user_question, raw_data = state["question"], state["query_result"]
            
            user_message = get_text2sql_format_user_prompt(user_question, raw_data)
            response = qwen.invoke([
                SystemMessage(content=TEXT2SQL_FORMAT_SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ])

            answer = response.content
            print(f"Query: {state.get('sql_candidate', '')}\nAnswer: {answer}")

        return {
            "formatted_result": answer,
            "data_results": [answer]
        }      

    def create_agent(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_vect_db_query", self.generate_vect_db_query)
        workflow.add_node("schema_db", self.retrieve_schema)
        workflow.add_node("example_db", self.retrieve_examples)
        workflow.add_node("evidence_db", self.retrieve_evidence)
        workflow.add_node("cell_value_db", self.retrieve_values)
        workflow.add_node("generate_query", self.generate_sql)
        workflow.add_node("syntax_checker", self.syntax_checker)
        workflow.add_node("semantic_checker", self.semantic_checker)
        workflow.add_node("format_result", self.format_result)

        workflow.add_edge(START, "generate_vect_db_query")

        workflow.add_edge("generate_vect_db_query", "schema_db")
        workflow.add_edge("generate_vect_db_query", "example_db")
        workflow.add_edge("generate_vect_db_query", "evidence_db")
        workflow.add_edge("generate_vect_db_query", "cell_value_db")

        workflow.add_edge("schema_db", "generate_query")
        workflow.add_edge("example_db", "generate_query")
        workflow.add_edge("evidence_db", "generate_query")
        workflow.add_edge("cell_value_db", "generate_query")
        workflow.add_edge("generate_query", "syntax_checker")

        workflow.add_conditional_edges(
            "syntax_checker",
            self.should_continue_syntax,
            {
                "syntax_checker": "syntax_checker",
                "semantic_checker": "semantic_checker"
            }
        )

        workflow.add_conditional_edges(
            "semantic_checker",
            self.check_semantic_modification,
            {
                "syntax_checker": "syntax_checker",
                "format_result": "format_result"
            }
        )

        workflow.add_edge("format_result", END)
        return workflow.compile()

    def run_agent(self, question: str, config: dict):
        result = self.agent.invoke({"question": question}, config=config)
        return result