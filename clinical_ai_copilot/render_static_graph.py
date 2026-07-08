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
    image = Image.new("RGB", (1600, 1350), "#f7f9fb")
    draw = ImageDraw.Draw(image)

    draw.text(
        (90, 55),
        "Clinical AI Copilot - Parallel LangGraph Workflow",
        fill="#101820",
        font=load_font(44, bold=True),
    )
    draw.text(
        (92, 112),
        "A planner fans out memory, imaging, and RAG branches; a reducer joins context before reasoning and safety review.",
        fill="#44515f",
        font=load_font(22),
    )

    boxes = {
        "planner": (590, 205, 1010, 315, "Planner", "Agent Orchestration", "#ffffff"),
        "memory": (70, 410, 430, 520, "Memory", "Case Context", "#f0e9ff"),
        "image": (520, 410, 880, 520, "Tool Calling", "Image Analysis", "#dff3ff"),
        "rag": (970, 410, 1330, 520, "RAG", "PubMed + Guidelines", "#e6f5df"),
        "radiomics": (520, 610, 880, 720, "Tool Calling", "Radiomics Agent", "#dff3ff"),
        "reducer": (590, 790, 1010, 905, "Reducer", "Evidence + Imaging Context", "#e9edf2"),
        "reasoning": (590, 945, 1010, 1060, "Multi-Agent AI", "Clinical Reasoning", "#fff1cc"),
        "evaluation": (70, 1140, 430, 1255, "LLM Evaluation", "AI Safety", "#ffe2df"),
        "human": (590, 1140, 1010, 1255, "Human-in-the-loop", "Review Gate", "#ede7ff"),
        "report": (1070, 1140, 1430, 1255, "Explainability", "Report Generator", "#e9edf2"),
    }

    for _, (x1, y1, x2, y2, title, subtitle, fill) in boxes.items():
        draw_box(draw, (x1, y1, x2, y2), title, subtitle, fill)

    draw.rounded_rectangle((590, 70, 1010, 155), radius=18, fill="#ffffff", outline="#24313f", width=2)
    draw.text((696, 90), "Case Inputs", fill="#101820", font=load_font(28, bold=True))
    draw.text((650, 123), "CT + Report + Patient Data", fill="#334150", font=load_font(18))
    draw_arrow(draw, (800, 155), (800, 205))

    draw_arrow(draw, (590, 260), (430, 465))
    draw_arrow(draw, (800, 315), (700, 410))
    draw_arrow(draw, (1010, 260), (970, 465))
    draw_arrow(draw, (700, 520), (700, 610))

    draw.line([(250, 520), (250, 755), (800, 755), (800, 790)], fill="#51606f", width=3)
    draw.polygon([(800, 790), (792, 777), (808, 777)], fill="#51606f")
    draw.line([(700, 720), (700, 755), (800, 755)], fill="#51606f", width=3)
    draw.line([(1150, 520), (1150, 755), (800, 755)], fill="#51606f", width=3)

    draw_arrow(draw, (800, 905), (800, 945))
    draw.line([(800, 1060), (800, 1095), (250, 1095), (250, 1140)], fill="#51606f", width=3)
    draw.polygon([(250, 1140), (242, 1127), (258, 1127)], fill="#51606f")
    draw_arrow(draw, (430, 1198), (590, 1198))
    draw_arrow(draw, (1010, 1198), (1070, 1198))

    draw.text((680, 765), "parallel reducer join", fill="#5b6875", font=load_font(18, bold=True))

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
