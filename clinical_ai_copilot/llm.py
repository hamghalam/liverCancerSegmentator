from __future__ import annotations

from typing import Any

from .config import CopilotConfig


class RuleBasedClinicalLLM:
    """Deterministic fallback used for demos, tests, and offline execution."""

    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "verify" in prompt_lower or "hallucination" in prompt_lower:
            return (
                "Verification result: no unsupported treatment recommendation should be treated "
                "as definitive. Require clinician review for guideline-dependent decisions."
            )
        if "resect" in prompt_lower or "surgery" in prompt_lower:
            return (
                "Clinical synthesis: surgical suitability is uncertain and depends on lesion "
                "distribution, liver reserve, performance status, and multidisciplinary review."
            )
        return (
            "Clinical synthesis: treatment response likelihood appears moderate when imaging "
            "burden is limited and confidence is acceptable, but evidence should be verified."
        )


def load_llm(config: CopilotConfig) -> Any:
    if config.mock_llm:
        return RuleBasedClinicalLLM()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Install copilot dependencies with `pip install -e .[copilot]` "
            "or set COPILOT_MOCK_LLM=1 for offline demo mode."
        ) from exc

    return ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key,
        temperature=0.1,
    )
