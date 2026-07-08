from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


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


def load_case_memory(case_id: str | None, memory_path: str | None) -> dict[str, Any]:
    if not case_id or not memory_path:
        return {"status": "not_available", "case_id": case_id, "prior_context": None}

    path = Path(memory_path)
    if not path.exists():
        return {"status": "empty", "case_id": case_id, "prior_context": None}

    with path.open("r", encoding="utf-8") as handle:
        memory = json.load(handle)

    return {
        "status": "loaded" if case_id in memory else "not_found",
        "case_id": case_id,
        "prior_context": memory.get(case_id),
        "memory_path": str(path),
    }


def upsert_case_memory(case_id: str | None, memory_path: str | None, summary: dict[str, Any]) -> None:
    if not case_id or not memory_path:
        return

    path = Path(memory_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    memory = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            memory = json.load(handle)

    memory[case_id] = summary
    with path.open("w", encoding="utf-8") as handle:
        json.dump(memory, handle, indent=2)


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


def search_pubmed(query: str, max_results: int = 5, timeout: int = 10) -> list[dict[str, Any]]:
    """Search PubMed through NCBI E-utilities and return RAG-ready snippets."""

    search_params = urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
    )
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}"
    with urlopen(search_url, timeout=timeout) as response:
        search_payload = json.loads(response.read().decode("utf-8"))

    pubmed_ids = search_payload.get("esearchresult", {}).get("idlist", [])
    if not pubmed_ids:
        return []

    summary_params = urlencode(
        {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),
            "retmode": "json",
        }
    )
    summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_params}"
    with urlopen(summary_url, timeout=timeout) as response:
        summary_payload = json.loads(response.read().decode("utf-8"))

    result = summary_payload.get("result", {})
    articles = []
    for pubmed_id in pubmed_ids:
        item = result.get(pubmed_id, {})
        if not item:
            continue
        articles.append(
            {
                "source": "pubmed",
                "pubmed_id": pubmed_id,
                "title": item.get("title", "Untitled PubMed article"),
                "snippet": item.get("fulljournalname", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                "supports": ["literature_search", "rag_context"],
            }
        )
    return articles


def retrieve_medical_evidence(
    query: str,
    guidelines: list[str],
    enable_pubmed: bool = True,
    pubmed_max_results: int = 5,
) -> list[dict[str, Any]]:
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
    if enable_pubmed:
        try:
            evidence.extend(search_pubmed(query=query, max_results=pubmed_max_results))
        except Exception as exc:
            evidence.append(
                {
                    "source": "pubmed_search_error",
                    "title": "PubMed retrieval unavailable",
                    "snippet": f"PubMed query prepared but search failed: {exc}",
                    "supports": ["literature_search", "rag_context"],
                }
            )
    else:
        evidence.append(
            {
                "source": "pubmed_search_disabled",
                "title": "PubMed retrieval disabled",
                "snippet": f"PubMed/RAG query prepared: {query}",
                "supports": ["literature_search", "rag_context"],
            }
        )
    return evidence


def evidence_supports_claim(claim: str, evidence: list[dict[str, Any]]) -> bool:
    claim_terms = {term for term in claim.lower().split() if len(term) > 5}
    evidence_text = " ".join(
        f"{item.get('title', '')} {item.get('snippet', '')}".lower() for item in evidence
    )
    return any(term in evidence_text for term in claim_terms)
