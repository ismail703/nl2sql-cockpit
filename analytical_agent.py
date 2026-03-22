from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage

from text2sql import Text2SQL
from models import qwen

from states import SubQuestionPlan, AnalyticalState
from prompts import ANALYTICAL_PLANNER_PROMPT

class AnalyticalAgent:
    def __init__(self):
        self.sql_agent = Text2SQL()        
        self.agent = self.create_workflow()

    def plan_queries(self, state: AnalyticalState):
        print(f"\n[INFO] Planning sub-queries for: '{state['original_question']}'")
        structured_planner = qwen.with_structured_output(SubQuestionPlan)
        
        plan = structured_planner.invoke([
            SystemMessage(content=ANALYTICAL_PLANNER_PROMPT),
            HumanMessage(content=state['original_question'])
        ])
        
        print(f"  -> Generated {len(plan.sub_questions)} sub-questions: {plan.sub_questions}")
        return {"sub_questions": plan.sub_questions}

    def dispatch_sub_queries(self, state: AnalyticalState):
        print(f"\n[INFO] Dispatching {len(state['sub_questions'])} sub-queries in parallel...")
        send_actions = []
        for q in state['sub_questions']:
            send_actions.append(Send("text2sql_agent", {"question": q}))
        return send_actions

    def create_workflow(self):
        workflow = StateGraph(AnalyticalState)
        workflow.add_node("plan_queries", self.plan_queries)        
        workflow.add_node("text2sql_agent", self.sql_agent.agent) 

        workflow.add_edge(START, "plan_queries")
        workflow.add_conditional_edges("plan_queries", self.dispatch_sub_queries, ["text2sql_agent"])
        workflow.add_edge("text2sql_agent", END)    
        
        return workflow.compile()

    def run(self, original_question: str):
        """Wrapper to invoke the agent from the Supervisor."""
        result = self.agent.invoke({
            "original_question": original_question, 
            "sub_questions": [], 
            "data_results": [],
        })
        return result