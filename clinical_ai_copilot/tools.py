from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_segmentation_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "status": "not_available",
            "message": "No segmentation metadata was supplied.",
        }

    metadata_path = Path(path)
    if not metadata_path.exists():
        return {
            "status": "missing",
            "message": f"Segmentation metadata file not found: {metadata_path}",
        }

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate_lesion_burden(metadata: dict[str, Any]) -> dict[str, Any]:
    volume_mm3 = metadata.get("tumor_volume_mm3")
    confidence = metadata.get("confidence_pairwise_dice")

    if volume_mm3 is None:
        burden = "unknown"
    elif volume_mm3 < 5_000:
        burden = "low"
    elif volume_mm3 < 30_000:
        burden = "moderate"
    else:
        burden = "high"

    if confidence is None or (isinstance(confidence, float) and math.isnan(confidence)):
        uncertainty = "unknown"
    elif confidence >= 0.85:
        uncertainty = "low"
    elif confidence >= 0.70:
        uncertainty = "moderate"
    else:
        uncertainty = "high"

    return {
        "tumor_volume_mm3": volume_mm3,
        "tumor_volume_ml": round(volume_mm3 / 1000, 3) if isinstance(volume_mm3, (int, float)) else None,
        "ensemble_confidence": confidence,
        "lesion_burden": burden,
        "segmentation_uncertainty": uncertainty,
    }


def extract_radiomics_stub(image_analysis: dict[str, Any], report: str) -> dict[str, Any]:
    burden = image_analysis.get("lesion_burden", "unknown")
    uncertainty = image_analysis.get("segmentation_uncertainty", "unknown")
    report_lower = report.lower()

    response_signal = "unknown"
    if "decrease" in report_lower or "partial response" in report_lower:
        response_signal = "favorable"
    elif "progress" in report_lower or "new lesion" in report_lower:
        response_signal = "unfavorable"
    elif burden in {"low", "moderate"} and uncertainty != "high":
        response_signal = "intermediate"

    return {
        "feature_family": [
            "shape_volume",
            "shape_surface",
            "first_order_intensity",
            "texture_glcm_glrlm",
        ],
        "extraction_status": "placeholder_ready_for_pyradiomics",
        "response_signal": response_signal,
        "notes": (
            "This module is designed to be replaced by PyRadiomics features from CT and mask "
            "pairs when imaging data is available."
        ),
    }


def retrieve_medical_evidence(query: str, guidelines: list[str]) -> list[dict[str, Any]]:
    local_guidelines = guidelines or [
        "Use multidisciplinary review for colorectal liver metastasis treatment planning.",
        "Assess resectability using lesion distribution, future liver remnant, extrahepatic disease, and performance status.",
        "Use systemic therapy response and imaging trends to guide treatment sequencing.",
    ]

    evidence = [
        {
            "source": "local_guideline_excerpt",
            "title": "CRLM multidisciplinary decision factors",
            "snippet": item,
            "supports": ["treatment_planning", "resectability", "human_review"],
        }
        for item in local_guidelines
    ]
    evidence.append(
        {
            "source": "pubmed_search_placeholder",
            "title": "PubMed retrieval hook",
            "snippet": f"Query prepared for PubMed/RAG retrieval: {query}",
            "supports": ["literature_search"],
        }
    )
    return evidence


def evidence_supports_claim(claim: str, evidence: list[dict[str, Any]]) -> bool:
    claim_terms = {term for term in claim.lower().split() if len(term) > 5}
    evidence_text = " ".join(
        f"{item.get('title', '')} {item.get('snippet', '')}".lower() for item in evidence
    )
    return any(term in evidence_text for term in claim_terms)
