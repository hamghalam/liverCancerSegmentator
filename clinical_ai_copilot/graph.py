from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .agents import (
    clinical_reasoning_agent,
    evidence_verification_agent,
    image_analysis_agent,
    literature_agent,
    radiomics_agent,
    report_generator_agent,
)
from .config import CopilotConfig
from .llm import load_llm
from .state import CopilotState


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
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required for the Clinical AI Copilot. "
            "Install it with `pip install -e .[copilot]`."
        ) from exc

    graph = StateGraph(CopilotState)
    graph.add_node("image_analysis", image_analysis_agent)
    graph.add_node("radiomics", radiomics_agent)
    graph.add_node("medical_literature", literature_agent)
    graph.add_node("clinical_reasoning", lambda state: clinical_reasoning_agent(state, llm))
    graph.add_node("evidence_verification", lambda state: evidence_verification_agent(state, llm))
    graph.add_node("report_generator", report_generator_agent)

    graph.set_entry_point("image_analysis")
    graph.add_edge("image_analysis", "radiomics")
    graph.add_edge("radiomics", "medical_literature")
    graph.add_edge("medical_literature", "clinical_reasoning")
    graph.add_edge("clinical_reasoning", "evidence_verification")
    graph.add_edge("evidence_verification", "report_generator")
    graph.add_edge("report_generator", END)
    return graph.compile()


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
