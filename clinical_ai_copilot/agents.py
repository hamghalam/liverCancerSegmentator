from __future__ import annotations

from typing import Any

from .state import CopilotState
from .tools import (
    estimate_lesion_burden,
    evidence_supports_claim,
    extract_radiomics_stub,
    load_case_memory,
    load_segmentation_metadata,
    retrieve_medical_evidence,
    upsert_case_memory,
)


def append_log(state: CopilotState, message: str) -> list[str]:
    return [message]


def planner_agent(state: CopilotState) -> CopilotState:
    plan = {
        "objective": state.get("clinical_question", "Clinical review of CRLM case."),
        "parallel_branches": [
            "Image Analysis -> Radiomics",
            "Medical Literature RAG",
            "Memory retrieval",
        ],
        "join_strategy": "Reducer merges imaging, radiomics, evidence, and memory before LLM reasoning.",
        "safety_gates": [
            "Evidence verification",
            "Human-in-the-loop review",
            "Research-use safety notice",
        ],
    }
    return {
        "execution_plan": plan,
        "audit_log": append_log(state, "Planner Agent created the case execution plan."),
    }


def memory_agent(state: CopilotState, memory_path: str | None = None) -> CopilotState:
    memory = load_case_memory(state.get("case_id"), memory_path)
    return {
        "memory_context": memory,
        "audit_log": append_log(state, "Memory Agent retrieved prior case context."),
    }


def image_analysis_agent(state: CopilotState) -> CopilotState:
    metadata = load_segmentation_metadata(state.get("segmentation_metadata_path"))
    analysis = estimate_lesion_burden(metadata)
    analysis["metadata_status"] = metadata.get("status", "loaded")
    analysis["case_id"] = state.get("case_id")
    return {
        "image_analysis": analysis,
        "audit_log": append_log(state, "Image Analysis Agent completed nnUNet metadata review."),
    }


def radiomics_agent(state: CopilotState) -> CopilotState:
    radiomics = extract_radiomics_stub(
        state.get("image_analysis", {}),
        state.get("radiology_report", ""),
    )
    return {
        "radiomics": radiomics,
        "audit_log": append_log(state, "Radiomics Agent generated feature and response summary."),
    }


def literature_agent(
    state: CopilotState,
    enable_pubmed: bool = True,
    pubmed_max_results: int = 5,
) -> CopilotState:
    query = " ".join(
        part
        for part in [
            "colorectal liver metastases",
            state.get("clinical_question", ""),
            state.get("radiology_report", "")[:300],
        ]
        if part
    )
    evidence = retrieve_medical_evidence(
        query=query,
        guidelines=state.get("guidelines", []),
        enable_pubmed=enable_pubmed,
        pubmed_max_results=pubmed_max_results,
    )
    return {
        "evidence": evidence,
        "audit_log": append_log(state, "Medical Literature Agent retrieved guideline/RAG evidence."),
    }


def clinical_reasoning_agent(state: CopilotState, llm: Any) -> CopilotState:
    image = state.get("image_analysis", {})
    radiomics = state.get("radiomics", {})
    evidence = state.get("evidence", [])
    reduced_context = state.get("reduced_context", {})
    patient = state.get("patient_profile", {})

    prompt = f"""
You are a clinical AI copilot for colorectal liver metastasis case review.
Do not provide autonomous medical decisions. Produce a cautious synthesis.

Patient profile: {patient}
Reduced context: {reduced_context}
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
        "clinical_reasoning": reasoning,
        "human_review_required": True,
        "audit_log": append_log(state, "Clinical Reasoning Agent synthesized case-level assessment."),
    }


def reducer_agent(state: CopilotState) -> CopilotState:
    image = state.get("image_analysis", {})
    radiomics = state.get("radiomics", {})
    evidence = state.get("evidence", [])
    memory = state.get("memory_context", {})
    plan = state.get("execution_plan", {})

    reduced_context = {
        "plan_objective": plan.get("objective"),
        "prior_case_context_available": bool(memory.get("prior_context")),
        "lesion_burden": image.get("lesion_burden"),
        "segmentation_uncertainty": image.get("segmentation_uncertainty"),
        "tumor_volume_ml": image.get("tumor_volume_ml"),
        "radiomics_response_signal": radiomics.get("response_signal"),
        "evidence_count": len(evidence),
        "top_evidence_titles": [item.get("title") for item in evidence[:3]],
        "safety_posture": "Require human review for all clinical conclusions.",
    }
    return {
        "reduced_context": reduced_context,
        "audit_log": append_log(state, "Reducer Agent merged parallel branch outputs for reasoning."),
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
        "verification": verification,
        "audit_log": append_log(state, "Evidence Verification Agent checked support and safety risks."),
    }


def human_review_agent(state: CopilotState, enable_interrupt: bool = False) -> CopilotState:
    verification = state.get("verification", {})
    review_packet = {
        "required": True,
        "reason": verification.get("status", "clinical_ai_requires_review"),
        "case_id": state.get("case_id"),
        "review_items": [
            "Confirm segmentation quality and uncertainty.",
            "Confirm evidence supports the clinical synthesis.",
            "Approve, edit, or reject the generated clinical report before use.",
        ],
    }

    reviewer_response = {
        "status": "pending",
        "reviewer": None,
        "notes": "Human review interrupt disabled. Report is marked pending review.",
    }
    if enable_interrupt:
        try:
            from langgraph.types import interrupt

            reviewer_response = interrupt(review_packet)
        except ImportError:
            reviewer_response["notes"] = "LangGraph interrupt API unavailable in this environment."

    return {
        "human_review": {
            "packet": review_packet,
            "response": reviewer_response,
        },
        "human_review_required": True,
        "audit_log": append_log(state, "Human-in-the-loop gate prepared clinician review packet."),
    }


def report_generator_agent(state: CopilotState, memory_path: str | None = None) -> CopilotState:
    plan = state.get("execution_plan", {})
    memory = state.get("memory_context", {})
    image = state.get("image_analysis", {})
    radiomics = state.get("radiomics", {})
    reduced_context = state.get("reduced_context", {})
    reasoning = state.get("clinical_reasoning", {})
    verification = state.get("verification", {})
    human_review = state.get("human_review", {})
    evidence = state.get("evidence", [])

    evidence_lines = "\n".join(
        f"- {item.get('title')}: {item.get('snippet')}" for item in evidence[:5]
    )
    report = f"""# Clinical AI Copilot Report

Case ID: {state.get("case_id", "unknown")}

## Planner Agent
- Objective: {plan.get("objective")}
- Join strategy: {plan.get("join_strategy")}

## Memory Agent
- Status: {memory.get("status")}
- Prior context available: {bool(memory.get("prior_context"))}

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

## Reducer Agent
- Reduced context: {reduced_context}

## Clinical Reasoning Agent
- Treatment response likelihood: {reasoning.get("treatment_response_likelihood")}
- Surgery suitability: {reasoning.get("surgery_suitability")}
- Summary: {reasoning.get("summary")}

## Evidence Verification Agent
- Status: {verification.get("status")}
- Hallucination warning: {verification.get("hallucination_warning")}
- Unsupported claims: {verification.get("unsupported_claims")}

## Human-in-the-loop Review
- Required: {state.get("human_review_required")}
- Review status: {human_review.get("response", {}).get("status")}
- Review note: {human_review.get("response", {}).get("notes")}

## Safety Notice
This report is a research copilot output. It is not a diagnosis, treatment recommendation,
or replacement for oncologist/radiologist review.
"""
    upsert_case_memory(
        state.get("case_id"),
        memory_path,
        {
            "treatment_response_likelihood": reasoning.get("treatment_response_likelihood"),
            "surgery_suitability": reasoning.get("surgery_suitability"),
            "segmentation_uncertainty": image.get("segmentation_uncertainty"),
            "human_review_status": human_review.get("response", {}).get("status"),
        },
    )
    return {
        "final_report": report,
        "audit_log": append_log(state, "Report Generator produced final clinical summary and updated memory."),
    }
