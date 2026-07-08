from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import load_case, run_copilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Clinical AI Copilot LangGraph workflow.")
    parser.add_argument(
        "--case",
        type=Path,
        default=Path(__file__).with_name("sample_case.json"),
        help="Path to a case JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clinical_ai_copilot_output") / "report.md",
        help="Where to write the generated report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full final state as JSON instead of the report only.",
    )
    args = parser.parse_args()

    result = run_copilot(load_case(args.case), output_path=args.output)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["final_report"])
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
