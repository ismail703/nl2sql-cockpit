import operator
from typing import List, TypedDict, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_tavily import TavilySearch

from models import llm, llama

from states import ResearchAgentState, SearchTask, SearchFinding, QueryPlan
from prompts import RESEARCH_QUERY_GENERATOR_PROMPT, REPORT_SYNTHESIS_PROMPT

class ResearchAgent:
    def __init__(self):
        self.search_tool = TavilySearch(max_results=5)
        self.agent = self.create_workflow()

    def generate_queries_node(self, state: ResearchAgentState):
        print("\n Generating research queries...")
        planner = llama.with_structured_output(QueryPlan)

        plan = planner.invoke([
            SystemMessage(content=RESEARCH_QUERY_GENERATOR_PROMPT),
            HumanMessage(content=state["analytical_finding"])
        ])

        queries: List[SearchTask] = (
            [{"query": q, "category": "explanation"} for q in plan.explanation_queries] +
            [{"query": q, "category": "competitor"} for q in plan.competitor_queries]
        )
        print(f"  -> {len(queries)} queries planned: {queries}")
        return {"queries": queries}

    def dispatch_search(self, state: ResearchAgentState):
        tasks = state.get("queries", [])
        if not tasks:
            print("No queries to search, skipping to synthesis.")
            return "synthesize_report"
        return [Send("search_node", task) for task in tasks]

    def search_node(self, state: SearchTask):
        query = state["query"]
        category = state["category"]
        print(f"Searching ({category}): {query} ...")
        try:
            raw_results = self.search_tool.invoke({"query": query})
            if isinstance(raw_results, list):
                formatted = "\n".join(
                    f"- {r.get('content', '')} (source: {r.get('url', 'n/a')})"
                    for r in raw_results
                )
            else:
                formatted = str(raw_results)
        except Exception as e:
            print(f"[WARN] Search failed for '{query}': {e}")
            formatted = "No results (search failed)."

        finding: SearchFinding = {"category": category, "query": query, "result": formatted}
        return {"search_results": [finding]}

    def synthesize_report_node(self, state: ResearchAgentState):
        print("\n Synthesizing final report...")
        results = state.get("search_results", [])

        explanation_block = "\n\n".join(
            f"Query: {r['query']}\n{r['result']}" for r in results if r["category"] == "explanation"
        ) or "No explanation-related search results."

        competitor_block = "\n\n".join(
            f"Query: {r['query']}\n{r['result']}" for r in results if r["category"] == "competitor"
        ) or "No competitor-related search results."

        context = (
            f"Analytical finding: {state['analytical_finding']}\n\n"
            f"=== Explanation research ===\n{explanation_block}\n\n"
            f"=== Competitor research ===\n{competitor_block}"
        )

        response = llm.invoke([
            SystemMessage(content=REPORT_SYNTHESIS_PROMPT),
            HumanMessage(content=context)
        ])

        print("Report generated.")
        return {"report": response.content}


    def create_workflow(self):
        workflow = StateGraph(ResearchAgentState)

        workflow.add_node("generate_queries", self.generate_queries_node)
        workflow.add_node("search_node", self.search_node)
        workflow.add_node("synthesize_report", self.synthesize_report_node)

        workflow.add_edge(START, "generate_queries")
        workflow.add_conditional_edges(
            "generate_queries",
            self.dispatch_search,
            ["search_node", "synthesize_report"]
        )
        workflow.add_edge("search_node", "synthesize_report")
        workflow.add_edge("synthesize_report", END)

        return workflow.compile()

    def run(self, analytical_finding: str) -> str:
        result = self.agent.invoke({"analytical_finding": analytical_finding})
        return result["report"]

