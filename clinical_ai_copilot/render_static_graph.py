from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#51606f", width=3)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    arrow_len = 16
    spread = 0.45
    left = (
        ex - arrow_len * math.cos(angle - spread),
        ey - arrow_len * math.sin(angle - spread),
    )
    right = (
        ex - arrow_len * math.cos(angle + spread),
        ey - arrow_len * math.sin(angle + spread),
    )
    draw.polygon([(ex, ey), left, right], fill="#51606f")


def draw_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    fill: str,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline="#24313f", width=2)
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(17)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    title_x = x1 + ((x2 - x1) - (title_box[2] - title_box[0])) // 2
    subtitle_x = x1 + ((x2 - x1) - (subtitle_box[2] - subtitle_box[0])) // 2
    draw.text((title_x, y1 + 22), title, fill="#101820", font=title_font)
    draw.text((subtitle_x, y1 + 58), subtitle, fill="#334150", font=subtitle_font)


def render_graph(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1280, 940), "#f7f9fb")
    draw = ImageDraw.Draw(image)

    draw.text(
        (60, 42),
        "Clinical AI Copilot - Parallel LangGraph Workflow",
        fill="#101820",
        font=load_font(34, bold=True),
    )
    draw.text(
        (62, 88),
        "Planner fans out memory, imaging, and RAG branches; reducer joins context before reasoning and safety review.",
        fill="#44515f",
        font=load_font(18),
    )

    boxes = {
        "planner": (465, 150, 815, 235, "Planner", "Agent Orchestration", "#ffffff"),
        "memory": (55, 330, 335, 410, "Memory", "Case Context", "#f0e9ff"),
        "image": (500, 330, 780, 410, "Tool Calling", "Image Analysis", "#dff3ff"),
        "rag": (945, 330, 1225, 410, "RAG", "PubMed + Guidelines", "#e6f5df"),
        "radiomics": (500, 485, 780, 565, "Tool Calling", "Radiomics", "#dff3ff"),
        "reducer": (465, 635, 815, 720, "Reducer", "Evidence + Imaging Context", "#e9edf2"),
        "reasoning": (465, 795, 815, 880, "Multi-Agent AI", "Clinical Reasoning", "#fff1cc"),
        "evaluation": (55, 795, 335, 880, "LLM Evaluation", "AI Safety", "#ffe2df"),
        "human": (875, 795, 1155, 880, "Human-in-the-loop", "Review Gate", "#ede7ff"),
    }

    for _, (x1, y1, x2, y2, title, subtitle, fill) in boxes.items():
        draw_box(draw, (x1, y1, x2, y2), title, subtitle, fill)

    draw.rounded_rectangle((465, 112, 815, 122), radius=5, fill="#24313f")
    draw.text((512, 115), "Case Inputs: CT + Report + Patient Data", fill="#334150", font=load_font(1))
    draw.text((485, 110), "Case Inputs: CT + Report + Patient Data", fill="#101820", font=load_font(18, bold=True))
    draw_arrow(draw, (640, 124), (640, 150))

    # Planner fan-out.
    draw.line([(640, 235), (640, 285), (195, 285), (195, 330)], fill="#51606f", width=3)
    draw.polygon([(195, 330), (187, 317), (203, 317)], fill="#51606f")
    draw.line([(640, 235), (640, 330)], fill="#51606f", width=3)
    draw.polygon([(640, 330), (632, 317), (648, 317)], fill="#51606f")
    draw.line([(640, 285), (1085, 285), (1085, 330)], fill="#51606f", width=3)
    draw.polygon([(1085, 330), (1077, 317), (1093, 317)], fill="#51606f")
    draw.text((570, 257), "parallel branches", fill="#5b6875", font=load_font(16, bold=True))

    # Image branch.
    draw_arrow(draw, (640, 410), (640, 485))

    # Reducer fan-in from memory, radiomics, and RAG.
    draw.line([(195, 410), (195, 602), (640, 602), (640, 635)], fill="#51606f", width=3)
    draw.polygon([(640, 635), (632, 622), (648, 622)], fill="#51606f")
    draw.line([(640, 565), (640, 635)], fill="#51606f", width=3)
    draw.polygon([(640, 635), (632, 622), (648, 622)], fill="#51606f")
    draw.line([(1085, 410), (1085, 602), (640, 602)], fill="#51606f", width=3)
    draw.text((548, 606), "reducer join", fill="#5b6875", font=load_font(16, bold=True))

    # Reasoning and safety gates.
    draw_arrow(draw, (640, 720), (640, 795))
    draw_arrow(draw, (465, 838), (335, 838))
    draw_arrow(draw, (815, 838), (875, 838))

    draw_box(draw, (465, 910, 815, 925), "Explainability", "Report Generator", "#e9edf2")
    draw_arrow(draw, (1015, 880), (815, 918))

    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static Clinical AI Copilot graph PNG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clinical_ai_copilot") / "assets" / "clinical_ai_copilot_graph.png",
    )
    args = parser.parse_args()
    render_graph(args.output)
    print(f"Static graph PNG written to: {args.output}")


if __name__ == "__main__":
    main()
