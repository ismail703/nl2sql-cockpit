import re
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from models import llm, qwen
from text2sql import Text2SQL

from states import SubQuestionPlan, SupervisorState, FeedbackEvaluation, Text2SQLRequests
from prompts import (
    ANALYTICAL_PLANNER_PROMPT,
    TASK_GENERATOR_PROMPT, 
    FEEDBACK_EVALUATOR_PROMPT, 
    REASONING_PROMPT,
    ANALYTICAL_REQUEST_GENERATOR_PROMPT
)


@tool
def calculate_percentage(part: float, total: float) -> str:
    """
    Calculate the percentage of a part out of a total amount.
    args:
        - part: (simple number) The portion or subset value.
        - total: (simple number) The total or reference value.
    """
    if total == 0:
        return "0%"
    percentage = (part / total) * 100
    return f"{percentage:.2f}%"

@tool
def compare_periods(current_value: float, previous_value: float) -> str:
    """Compare values between two periods and calculate growth/decline percentage."""
    if previous_value == 0:
        return "N/A (Previous value was 0)"
    diff = current_value - previous_value
    growth_rate = (diff / previous_value) * 100
    direction = "growth" if diff > 0 else "decline" if diff < 0 else "no change"
    return f"A {direction} of {abs(growth_rate):.2f}% (Absolute difference: {diff:+.2f})"


class SupervisorAgent:
    def __init__(self, checkpointer: PostgresSaver = None):        
        self.sql_agent = Text2SQL()         
        self.tools = [calculate_percentage, compare_periods]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.agent = self.create_workflow(checkpointer)

    def generate_task_node(self, state: SupervisorState):        
        print("\n[INFO] Supervisor generating task description...")
        user_input = state["messages"][-1].content                

        # history_msgs = "\n".join([msg.content for msg in state.get('messages', [])])        
        response = qwen.invoke([
            SystemMessage(content=TASK_GENERATOR_PROMPT),# + "\nMessage History:\n" + history_msgs),
            HumanMessage(content=user_input)
        ])
        
        cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        print(f"Generated Task:\n{cleaned_content}")
        
        return {"task_description": cleaned_content}

    def human_review_node(self, state: SupervisorState):
        print("\n[INFO] Pausing for human review...")
        
        user_feedback = interrupt({
            "status": "awaiting_approval",
            "message": "Please review the task description before I query the database.",
            "task_description": state["task_description"]
        })

        print(f"\n[INFO] Received user feedback: {user_feedback}")
        evaluator = llm.with_structured_output(FeedbackEvaluation)
        
        user_prompt = f"""
        Original Task Description: {state['task_description']}
        User Feedback: {user_feedback}
        """
        
        evaluation = evaluator.invoke([
            SystemMessage(content=FEEDBACK_EVALUATOR_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        
        if evaluation.is_approved:
            print("[INFO] LLM Evaluator: Task approved as-is.")
            return {"task_description": state["task_description"]}
        else:
            print(f"[INFO] LLM Evaluator: Task updated based on feedback.\nNew Task: {evaluation.updated_task_description}")
            return {"task_description": evaluation.updated_task_description}
    
    def plan_queries(self, state: SupervisorState):
        print(f"\n[INFO] Planning sub-queries ....")
        structured_planner = qwen.with_structured_output(SubQuestionPlan)
        
        plan = structured_planner.invoke([
            SystemMessage(content=ANALYTICAL_PLANNER_PROMPT),
            HumanMessage(content=state['task_description'])
        ])
        
        print(f"  -> Generated {len(plan.sub_questions)} sub-questions: {plan.sub_questions}")
        return {"sub_questions": plan.sub_questions}
    
    def check_conversation_hist(self, state: SupervisorState):
        print("\n[INFO] Checking conversation history...")        
        structured_llm = llm.with_structured_output(
            Text2SQLRequests,
            method="json_mode"
        )

        current_sub_questions = state.get("sub_questions", [])
        history_msgs = "\n".join([msg.content for msg in state.get('messages', [])])        
        response = structured_llm.invoke([
            SystemMessage(content=ANALYTICAL_REQUEST_GENERATOR_PROMPT + "\nMessage History:\n" + history_msgs),
            HumanMessage(content=" ".join(current_sub_questions))
        ])

        print("Response : ", response.data)
        
        return {"sub_questions": response.sub_questions, "data_results": response.data}      

    def dispatch_sub_queries(self, state: SupervisorState):
            sub_questions = state.get("sub_questions", [])
            
            if len(sub_questions) > 0:
                print(f"\n[INFO] Dispatching {len(sub_questions)} sub-queries in parallel...")
                send_actions = []
                for q in sub_questions:
                    send_actions.append(Send("text2sql_agent", {"question": q}))
                return send_actions
            else:
                print("\n[INFO] No sub-queries to dispatch. Routing straight to reasoning...")
                return "reasoning"   

    def reasoning_and_calc_node(self, state: SupervisorState):
        print("\n[INFO] Supervisor performing final reasoning and calculations...")

        results_list = state.get("data_results", [])
        print(f"Data results to reason over: {results_list}")
        data_context = "\n".join(results_list)
        
        context = (            
            f"Executed Task: {state['task_description']}\n"
            f"Database Results: {data_context}"
        )
            
        messages_to_pass = [
            SystemMessage(content=REASONING_PROMPT), 
            HumanMessage(content=context)
        ] + state.get("messages", [])
        
        response = self.llm_with_tools.invoke(messages_to_pass)
        if response.tool_calls:
            print(f"\n[DEBUG] LLM is calling tools: {response.tool_calls}")
        else:
            print(f"\n[DEBUG] LLM Final Answer: {response.content}")
        
        return {"messages": [response]}

    def create_workflow(self, checkpointer):
        workflow = StateGraph(SupervisorState)

        workflow.add_node("generate_task", self.generate_task_node)
        workflow.add_node("human_review", self.human_review_node)
        workflow.add_node("plan_queries", self.plan_queries)
        workflow.add_node("check_conversation_hist", self.check_conversation_hist)
        workflow.add_node("text2sql_agent", self.sql_agent.agent) 
        workflow.add_node("reasoning", self.reasoning_and_calc_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "generate_task")
        workflow.add_edge("generate_task", "human_review")
        workflow.add_edge("human_review", "plan_queries")
        workflow.add_edge("plan_queries", "check_conversation_hist")       
        workflow.add_conditional_edges("check_conversation_hist", self.dispatch_sub_queries, ["text2sql_agent", "reasoning"])
        workflow.add_edge("text2sql_agent", "reasoning")    
        
        workflow.add_conditional_edges(
            "reasoning", 
            tools_condition, 
            {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "reasoning")

        return workflow.compile(checkpointer=checkpointer)