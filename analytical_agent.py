from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START 
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Send

from states import AnalyticalState, SubQuestionPlan
from prompts import (
    ANALYTICAL_PLANNER_SYSTEM_PROMPT,
    ANALYTICAL_REASONING_SYSTEM_PROMPT,
    get_analytical_reasoning_user_prompt
)
from models import llm, qwen
from text2sql import Text2SQL
from dotenv import load_dotenv
import os 

load_dotenv()
DB_URI = os.getenv("DB_URI")

@tool
def calculate_percentage(part: float, total: float) -> str:
    """
    Calculate the percentage of a part out of a total amount.
    Use this when the user asks for a share, percentage, or ratio.
    """
    if total == 0:
        return "0%"
    percentage = (part / total) * 100
    return f"{percentage:.2f}%"

@tool
def compare_periods(current_value: float, previous_value: float) -> str:
    """
    Compare values between two periods (e.g., MTD, YTD, Month 4 vs Month 3, Year over Year) 
    and calculate the growth or decline percentage.
    Use this whenever the user asks for a comparison, variance, growth rate, or difference 
    between a current/target metric and a previous/baseline metric.
    """
    if previous_value == 0:
        return "N/A (Previous value was 0)"
    
    diff = current_value - previous_value
    growth_rate = (diff / previous_value) * 100
    
    direction = "growth" if diff > 0 else "decline" if diff < 0 else "no change"
    return f"A {direction} of {abs(growth_rate):.2f}% (Absolute difference: {diff:+.2f})"


class AnalyticalAgent:
    def __init__(self, checkpointer: PostgresSaver = None):
        self.tools = [calculate_percentage, compare_periods]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.sql_agent = Text2SQL()        
        self.agent = self.create_workflow(checkpointer=checkpointer)

    def plan_queries(self, state: AnalyticalState):
        print(f"\n[INFO] Planning sub-queries for the question: '{state['messages'][-1].content}'")
        
        structured_planner = qwen.with_structured_output(SubQuestionPlan)
        
        plan = structured_planner.invoke(
            [SystemMessage(content=ANALYTICAL_PLANNER_SYSTEM_PROMPT)] + state.get("messages", [])
        )
        
        print(f"  -> Generated {len(plan.sub_questions)}\n sub-questions: {plan.sub_questions}")
        return {"sub_questions": plan.sub_questions}

    def dispatch_sub_queries(self, state: AnalyticalState):
        print(f"\n[INFO] Dispatching {len(state['sub_questions'])} sub-queries in parallel...")
        
        send_actions = []
        for q in state['sub_questions']:
            send_actions.append(Send("text2sql_agent", {"question": q}))
            
        return send_actions   

    def reasoning_node(self, state: AnalyticalState):
        results_list = state.get("data_results", [])
        data_context = "\n".join([f"- Q: {item['question']}\n  A: {item['answer']}" for item in results_list])

        system_prompt = SystemMessage(content=ANALYTICAL_REASONING_SYSTEM_PROMPT)            
        user_prompt_str = get_analytical_reasoning_user_prompt(state['messages'][-1].content, data_context)
        user_prompt = HumanMessage(content=user_prompt_str)

        messages = [system_prompt, user_prompt]
        
        response = self.llm_with_tools.invoke(state.get("messages", []) + messages)
        return {"messages": [response]}

    def create_workflow(self, checkpointer=None):     
        workflow = StateGraph(AnalyticalState)

        workflow.add_node("plan_queries", self.plan_queries)
        workflow.add_node("reasoning", self.reasoning_node)
        workflow.add_node("tools", ToolNode(self.tools))         
        workflow.add_node("text2sql_agent", self.sql_agent.agent) 
        
        workflow.add_edge(START, "plan_queries")            
        workflow.add_conditional_edges("plan_queries", self.dispatch_sub_queries, ["text2sql_agent"])
        workflow.add_edge("text2sql_agent", "reasoning")            
        workflow.add_conditional_edges("reasoning", tools_condition, {"tools": "tools", "__end__": END})
        workflow.add_edge("tools", "reasoning")

        return workflow.compile(checkpointer=checkpointer)

    def chat(self, user_input: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}            
        result = self.agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        return result["messages"][-1].content
    
if __name__ == "__main__":
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup() 
        bot = AnalyticalAgent(checkpointer)

        while True:
            user_message = input("You: ")
            if user_message.lower() in ['exit', 'quit']:
                break
            
            response = bot.chat(user_message, thread_id="7")
            print(f"\nBot: {response}\n")    