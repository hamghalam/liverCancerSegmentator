from __future__ import annotations

from typing import Any

from .state import CopilotState
from .tools import (
    estimate_lesion_burden,
    evidence_supports_claim,
    extract_radiomics_stub,
    load_segmentation_metadata,
    retrieve_medical_evidence,
)


def append_log(state: CopilotState, message: str) -> list[str]:
    return [*state.get("audit_log", []), message]


def image_analysis_agent(state: CopilotState) -> CopilotState:
    metadata = load_segmentation_metadata(state.get("segmentation_metadata_path"))
    analysis = estimate_lesion_burden(metadata)
    analysis["metadata_status"] = metadata.get("status", "loaded")
    analysis["case_id"] = state.get("case_id")
    return {
        **state,
        "image_analysis": analysis,
        "audit_log": append_log(state, "Image Analysis Agent completed nnUNet metadata review."),
    }


def radiomics_agent(state: CopilotState) -> CopilotState:
    radiomics = extract_radiomics_stub(
        state.get("image_analysis", {}),
        state.get("radiology_report", ""),
    )
    return {
        **state,
        "radiomics": radiomics,
        "audit_log": append_log(state, "Radiomics Agent generated feature and response summary."),
    }


def literature_agent(state: CopilotState) -> CopilotState:
    query = " ".join(
        part
        for part in [
            "colorectal liver metastases",
            state.get("clinical_question", ""),
            state.get("radiology_report", "")[:300],
        ]
        if part
    )
    evidence = retrieve_medical_evidence(query=query, guidelines=state.get("guidelines", []))
    return {
        **state,
        "evidence": evidence,
        "audit_log": append_log(state, "Medical Literature Agent retrieved guideline/RAG evidence."),
    }


def clinical_reasoning_agent(state: CopilotState, llm: Any) -> CopilotState:
    image = state.get("image_analysis", {})
    radiomics = state.get("radiomics", {})
    evidence = state.get("evidence", [])
    patient = state.get("patient_profile", {})

    prompt = f"""
You are a clinical AI copilot for colorectal liver metastasis case review.
Do not provide autonomous medical decisions. Produce a cautious synthesis.

Patient profile: {patient}
Image analysis: {image}
Radiomics summary: {radiomics}
Evidence snippets: {evidence}
Clinical question: {state.get("clinical_question", "")}
"""
    response = llm.invoke(prompt)
    text = getattr(response, "content", response)

    uncertainty = image.get("segmentation_uncertainty", "unknown")
    response_signal = radiomics.get("response_signal", "unknown")
    if response_signal == "favorable" and uncertainty != "high":
        treatment_response = "moderate_to_favorable"
    elif response_signal == "unfavorable" or uncertainty == "high":
        treatment_response = "uncertain_or_concerning"
    else:
        treatment_response = "moderate"

    reasoning = {
        "summary": str(text),
        "treatment_response_likelihood": treatment_response,
        "surgery_suitability": "requires_multidisciplinary_review",
        "human_review_required": True,
        "key_inputs": {
            "lesion_burden": image.get("lesion_burden"),
            "segmentation_uncertainty": uncertainty,
            "radiomics_response_signal": response_signal,
        },
    }
    return {
        **state,
        "clinical_reasoning": reasoning,
        "human_review_required": True,
        "audit_log": append_log(state, "Clinical Reasoning Agent synthesized case-level assessment."),
    }


def evidence_verification_agent(state: CopilotState, llm: Any) -> CopilotState:
    reasoning = state.get("clinical_reasoning", {})
    evidence = state.get("evidence", [])
    claims = [
        reasoning.get("treatment_response_likelihood", ""),
        reasoning.get("surgery_suitability", ""),
        reasoning.get("summary", ""),
    ]
    unsupported_claims = [
        claim for claim in claims if claim and not evidence_supports_claim(str(claim), evidence)
    ]
    prompt = f"""
Verify this clinical AI reasoning for unsupported claims or hallucination risk.
Reasoning: {reasoning}
Evidence: {evidence}
Unsupported lexical matches: {unsupported_claims}
"""
    response = llm.invoke(prompt)
    text = getattr(response, "content", response)
    verification = {
        "status": "needs_human_review" if unsupported_claims else "evidence_aligned_with_caution",
        "unsupported_claims": unsupported_claims,
        "hallucination_warning": bool(unsupported_claims),
        "verifier_note": str(text),
    }
    return {
        **state,
        "verification": verification,
        "audit_log": append_log(state, "Evidence Verification Agent checked support and safety risks."),
    }


def report_generator_agent(state: CopilotState) -> CopilotState:
    image = state.get("image_analysis", {})
    radiomics = state.get("radiomics", {})
    reasoning = state.get("clinical_reasoning", {})
    verification = state.get("verification", {})
    evidence = state.get("evidence", [])

    evidence_lines = "\n".join(
        f"- {item.get('title')}: {item.get('snippet')}" for item in evidence[:5]
    )
    report = f"""# Clinical AI Copilot Report

Case ID: {state.get("case_id", "unknown")}

## Image Analysis Agent
- Tumor volume: {image.get("tumor_volume_ml")} mL
- Lesion burden: {image.get("lesion_burden")}
- Ensemble confidence: {image.get("ensemble_confidence")}
- Segmentation uncertainty: {image.get("segmentation_uncertainty")}

## Radiomics Agent
- Extraction status: {radiomics.get("extraction_status")}
- Response signal: {radiomics.get("response_signal")}
- Feature families: {", ".join(radiomics.get("feature_family", []))}

## Medical Literature Agent
{evidence_lines}

## Clinical Reasoning Agent
- Treatment response likelihood: {reasoning.get("treatment_response_likelihood")}
- Surgery suitability: {reasoning.get("surgery_suitability")}
- Summary: {reasoning.get("summary")}

## Evidence Verification Agent
- Status: {verification.get("status")}
- Hallucination warning: {verification.get("hallucination_warning")}
- Unsupported claims: {verification.get("unsupported_claims")}

## Safety Notice
This report is a research copilot output. It is not a diagnosis, treatment recommendation,
or replacement for oncologist/radiologist review.
"""
    return {
        **state,
        "final_report": report,
        "audit_log": append_log(state, "Report Generator produced final clinical summary."),
    }
