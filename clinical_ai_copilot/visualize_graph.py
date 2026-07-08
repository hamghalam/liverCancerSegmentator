from __future__ import annotations

import argparse
from pathlib import Path

from .graph import save_graph_mermaid, save_graph_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Clinical AI Copilot LangGraph diagram.")
    parser.add_argument("--output", type=Path, default=None, help="Graph output path.")
    parser.add_argument(
        "--format",
        choices=("mermaid", "png"),
        default="mermaid",
        help="Export Mermaid text locally or render a PNG through LangGraph/Mermaid.",
    )
    args = parser.parse_args()

    if args.output is None:
        suffix = "png" if args.format == "png" else "mmd"
        args.output = Path("clinical_ai_copilot_output") / f"langgraph.{suffix}"

    output = save_graph_png(args.output) if args.format == "png" else save_graph_mermaid(args.output)
    print(f"LangGraph {args.format} written to: {output}")
    print("\nNotebook display snippet:")
    print("from IPython.display import Image, display")
    print("from clinical_ai_copilot.graph import build_copilot_graph")
    print("react_graph = build_copilot_graph()")
    print("display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))")


if __name__ == "__main__":
    main()
