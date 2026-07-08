from __future__ import annotations

import argparse
from pathlib import Path

from .graph import build_copilot_graph, save_graph_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Clinical AI Copilot LangGraph diagram.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clinical_ai_copilot_output") / "langgraph.png",
        help="PNG output path.",
    )
    args = parser.parse_args()

    output = save_graph_png(args.output)
    print(f"LangGraph diagram written to: {output}")
    print("\nNotebook display snippet:")
    print("from IPython.display import Image, display")
    print("from clinical_ai_copilot.graph import build_copilot_graph")
    print("react_graph = build_copilot_graph()")
    print("display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))")


if __name__ == "__main__":
    main()
