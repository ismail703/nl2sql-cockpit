from datetime import datetime
import re
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from models import llm, llama

from agents.text2sql import Text2SQL
from agents.research_agent import ResearchAgent

from states import SupervisorState, FeedbackEvaluation, Text2SQLRequests, RouteDecision, MemoryReconciliation
from prompts import ( 
    ANALYTICAL_PLANNER_AND_CHECKER_PROMPT,
    TASK_GENERATOR_PROMPT,
    FEEDBACK_EVALUATOR_PROMPT,
    REASONING_PROMPT,
    LESSON_EXTRACTOR_PROMPT,
    ENTRY_ROUTER_PROMPT,
    MEMORY_RECONCILER_PROMPT
)

from memory_store import LongTermMemory


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
        self.research_agent = ResearchAgent()
        self.memory = LongTermMemory()
        self.tools = [calculate_percentage, compare_periods]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.agent = self.create_workflow(checkpointer)

    def entry_router_node(self, state: SupervisorState):
        print("\n[INFO] Routing incoming message...")
        user_input = state["messages"][-1].content

        router = llm.with_structured_output(RouteDecision)
        decision = router.invoke([
            SystemMessage(content=ENTRY_ROUTER_PROMPT),
            HumanMessage(content=user_input)
        ])

        print(f"[INFO] Router decision: {decision.route}")

        result = {"route_decision": decision.route}

        if decision.route == "feedback":
            result["correction_notes"] = user_input

        return result

    def route_entry(self, state: SupervisorState):
        decision = state.get("route_decision")
        if decision == "feedback":
            return "store_memory"
        elif decision == "analytical":
            return "recall_memory"
        else:
            return "greeting"

    def greeting_node(self, state: SupervisorState):
        print("\n[INFO] Message unrelated to data/feedback — responding directly...")
        intro = (
            "Hello! I am your inwi assistant for data analysis and insights. "
            "I can query the database to answer your analytical questions (metrics, period comparisons, top-up trends, etc.), "
            "and I also take your feedback into account to improve my future responses."
            " Feel free to ask me a question about your data, or to point out a correction."
        )
        return {"messages": [AIMessage(content=intro)]}

    def recall_memory_node(self, state: SupervisorState):
        print("\n[INFO] Recalling relevant lessons...")
        user_input = state["messages"][-1].content
        memory_context = self.memory.recall(user_input, k=3)

        if memory_context:
            print(f"[INFO] Recalled lessons:\n{memory_context}")
        else:
            print("[INFO] No relevant lessons found.")

        return {"memory_context": memory_context}

    def generate_task_node(self, state: SupervisorState):        
        print("\n[INFO] Supervisor generating task description...")
        user_input = state["messages"][-1].content
        memory_context = state.get("memory_context", "")

        system_content = TASK_GENERATOR_PROMPT.format(current_date=datetime.now().strftime("%B %d, %Y"))
        if memory_context:
            system_content += (
                "\n\nPrevious user feedback and lessons learned to consider (apply only if genuinely relevant and related to the task; do not force a connection):\n"
                f"{memory_context}"
            )

        response = llama.invoke([
            SystemMessage(content=system_content),
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
        evaluator = llama.with_structured_output(FeedbackEvaluation)
        
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
            return {
                "task_description": evaluation.updated_task_description,
                "correction_notes": user_feedback,
            }   
         
    def plan_and_check_queries(self, state: SupervisorState):
        print("\n[INFO] Planning sub-queries and checking conversation history...")
        
        structured_llm = llm.with_structured_output(
            Text2SQLRequests,
            method="json_mode"
        )

        history_msgs = "\n".join([msg.content for msg in state.get('messages', [])])

        response = structured_llm.invoke([
            SystemMessage(content=ANALYTICAL_PLANNER_AND_CHECKER_PROMPT + "\nConversation History:\n" + history_msgs),
            HumanMessage(content=state['task_description'])
        ])

        print(f"  -> Resolved data: {response.data}")
        print(f"  -> Remaining sub-questions: {response.sub_questions}")

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
        print(f"Context:\n{context}")
            
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
    
    def store_memory_node(self, state: SupervisorState, config: RunnableConfig):
        print("\n[INFO] Extracting lesson for long-term memory...")

        correction_notes = state.get("correction_notes", "")
        chat_id = config.get("configurable", {}).get("thread_id")

        if not correction_notes:
            print("[INFO] No feedback to process.")
            return {}

        extraction_input = f"\n\nHuman feedback:\n{correction_notes}"

        try:
            response = llm.invoke([
                SystemMessage(content=LESSON_EXTRACTOR_PROMPT),
                HumanMessage(content=extraction_input)
            ])
            candidate_lesson = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

            if not candidate_lesson or candidate_lesson.upper() == "NONE":
                print("[INFO] No lesson worth storing for this task.")
                return {}

            similar = self.memory.recall_with_ids(candidate_lesson, k=3)
            similar_block = "\n".join(f"[{item['id']}] {item['lesson']}" for item in similar) or "(none)"
            print(f"[INFO] Candidate lesson:\n{candidate_lesson}\n\nSimilar lessons:\n{similar_block}")

            reconciler = llm.with_structured_output(MemoryReconciliation)
            decision = reconciler.invoke([
                SystemMessage(content=MEMORY_RECONCILER_PROMPT),
                HumanMessage(content=(
                    f"NEW candidate lesson:\n{candidate_lesson}\n\n"
                    f"EXISTING similar lessons:\n{similar_block}"
                ))
            ])

            print(f"[INFO] Reconciliation decision: {decision.action}")

            if decision.action == "add":
                entry_id = self.memory.add_lesson(lesson=candidate_lesson, chat_id=chat_id)
                print(f"[INFO] Lesson added: {entry_id} -> {candidate_lesson}")

            elif decision.action == "update" and decision.target_id and decision.final_lesson:
                self.memory.update_lesson(decision.target_id, decision.final_lesson)
                print(f"[INFO] Lesson updated: {decision.target_id} -> {decision.final_lesson}")

            elif decision.action == "delete" and decision.target_id:
                self.memory.delete_lesson(decision.target_id)
                print(f"[INFO] Lesson deleted: {decision.target_id}")

            else:
                print("[INFO] Skipped — duplicate of existing lesson.")

        except Exception as e:
            print(f"[WARN] Failed to extract/reconcile/store lesson: {e}")

        return {}

    def research_node(self, state: SupervisorState):
        print("\n Handing off final finding to Research Agent...")
        finding = state["messages"][-1].content
        report = self.research_agent.run(finding)
        return {"messages": [AIMessage(content=report)]}

    def create_workflow(self, checkpointer):
        workflow = StateGraph(SupervisorState)

        workflow.add_node("entry_router", self.entry_router_node)
        workflow.add_node("greeting", self.greeting_node)
        workflow.add_node("recall_memory", self.recall_memory_node)
        workflow.add_node("generate_task", self.generate_task_node)
        workflow.add_node("human_review", self.human_review_node)
        workflow.add_node("plan_queries", self.plan_and_check_queries)
        workflow.add_node("text2sql_agent", self.sql_agent.agent)
        workflow.add_node("reasoning", self.reasoning_and_calc_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("store_memory", self.store_memory_node)

        workflow.add_edge(START, "entry_router")
        workflow.add_conditional_edges(
            "entry_router",
            self.route_entry,
            {
                "store_memory": "store_memory",
                "recall_memory": "recall_memory",
                "greeting": "greeting",
            },
        )
        workflow.add_edge("greeting", END)

        workflow.add_edge("recall_memory", "generate_task")
        workflow.add_edge("generate_task", "human_review")
        workflow.add_edge("human_review", "plan_queries")
        workflow.add_conditional_edges("plan_queries", self.dispatch_sub_queries, ["text2sql_agent", "reasoning"])
        workflow.add_edge("text2sql_agent", "reasoning")

        workflow.add_conditional_edges(
            "reasoning",
            tools_condition,
            {"tools": "tools", "__end__": 'research'}
        )
        workflow.add_edge("tools", "reasoning")
        workflow.add_edge("research", "store_memory")
        workflow.add_edge("store_memory", END)

        return workflow.compile(checkpointer=checkpointer)