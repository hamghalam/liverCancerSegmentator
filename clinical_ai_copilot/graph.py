from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .agents import (
    clinical_reasoning_agent,
    evidence_verification_agent,
    human_review_agent,
    image_analysis_agent,
    literature_agent,
    memory_agent,
    planner_agent,
    radiomics_agent,
    reducer_agent,
    report_generator_agent,
)
from .config import CopilotConfig
from .llm import load_llm
from .state import CopilotState


NODE_PLANNER = "Planner - Agent Orchestration"
NODE_MEMORY = "Memory - Case Context"
NODE_IMAGE = "Tool Calling - Image Analysis"
NODE_RADIOMICS = "Tool Calling - Radiomics"
NODE_RAG = "RAG - PubMed + Guidelines"
NODE_REDUCER = "Reducer - Evidence + Imaging Context"
NODE_REASONING = "Multi-Agent AI - Clinical Reasoning"
NODE_EVALUATION = "LLM Evaluation + AI Safety"
NODE_HUMAN = "Human-in-the-loop Review"
NODE_REPORT = "Explainability - Report Generator"


def configure_langsmith(config: CopilotConfig) -> None:
    if not config.enable_langsmith:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", config.langsmith_project)


def build_copilot_graph(config: CopilotConfig | None = None) -> Any:
    config = config or CopilotConfig.from_env()
    configure_langsmith(config)
    llm = load_llm(config)

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required for the Clinical AI Copilot. "
            "Install it with `pip install -e .[copilot]`."
        ) from exc

    graph = StateGraph(CopilotState)
    graph.add_node(NODE_PLANNER, planner_agent)
    graph.add_node(NODE_MEMORY, lambda state: memory_agent(state, memory_path=config.memory_path))
    graph.add_node(NODE_IMAGE, image_analysis_agent)
    graph.add_node(NODE_RADIOMICS, radiomics_agent)
    graph.add_node(
        NODE_RAG,
        lambda state: literature_agent(
            state,
            enable_pubmed=config.enable_pubmed,
            pubmed_max_results=config.pubmed_max_results,
        ),
    )
    graph.add_node(NODE_REASONING, lambda state: clinical_reasoning_agent(state, llm))
    graph.add_node(NODE_REDUCER, reducer_agent)
    graph.add_node(NODE_EVALUATION, lambda state: evidence_verification_agent(state, llm))
    graph.add_node(
        NODE_HUMAN,
        lambda state: human_review_agent(
            state,
            enable_interrupt=config.enable_human_interrupt,
        ),
    )
    graph.add_node(NODE_REPORT, lambda state: report_generator_agent(state, memory_path=config.memory_path))

    graph.add_edge(START, NODE_PLANNER)
    graph.add_edge(NODE_PLANNER, NODE_MEMORY)
    graph.add_edge(NODE_PLANNER, NODE_IMAGE)
    graph.add_edge(NODE_PLANNER, NODE_RAG)
    graph.add_edge(NODE_IMAGE, NODE_RADIOMICS)
    graph.add_edge([NODE_MEMORY, NODE_RADIOMICS, NODE_RAG], NODE_REDUCER)
    graph.add_edge(NODE_REDUCER, NODE_REASONING)
    graph.add_edge(NODE_REASONING, NODE_EVALUATION)
    graph.add_edge(NODE_EVALUATION, NODE_HUMAN)
    graph.add_edge(NODE_HUMAN, NODE_REPORT)
    graph.add_edge(NODE_REPORT, END)
    return graph.compile()


def save_graph_png(output_path: str | Path = "clinical_ai_copilot_output/langgraph.png") -> Path:
    app = build_copilot_graph()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    png = app.get_graph(xray=True).draw_mermaid_png()
    output.write_bytes(png)
    return output


def save_graph_mermaid(output_path: str | Path = "clinical_ai_copilot_output/langgraph.mmd") -> Path:
    app = build_copilot_graph()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mermaid = app.get_graph(xray=True).draw_mermaid()
    output.write_text(mermaid, encoding="utf-8")
    return output


def run_copilot(case: CopilotState, output_path: str | Path | None = None) -> CopilotState:
    app = build_copilot_graph()
    result = app.invoke(case)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result["final_report"], encoding="utf-8")

    return result


def load_case(path: str | Path) -> CopilotState:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
