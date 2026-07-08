from __future__ import annotations

from typing import Any, Literal, TypedDict


RiskLevel = Literal["low", "moderate", "high", "unknown"]


class CopilotState(TypedDict, total=False):
    """Shared state passed between LangGraph agents."""

    case_id: str
    ct_path: str | None
    segmentation_metadata_path: str | None
    radiology_report: str
    patient_profile: dict[str, Any]
    clinical_question: str
    guidelines: list[str]
    human_review_required: bool

    image_analysis: dict[str, Any]
    radiomics: dict[str, Any]
    evidence: list[dict[str, Any]]
    clinical_reasoning: dict[str, Any]
    verification: dict[str, Any]
    human_review: dict[str, Any]
    final_report: str
    audit_log: list[str]
