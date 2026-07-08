from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotConfig:
    """Runtime configuration loaded from environment variables."""

    google_api_key: str | None
    model_name: str
    enable_langsmith: bool
    langsmith_project: str
    mock_llm: bool
    enable_pubmed: bool
    pubmed_max_results: int
    enable_human_interrupt: bool

    @classmethod
    def from_env(cls) -> "CopilotConfig":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        return cls(
            google_api_key=google_api_key,
            model_name=os.getenv("COPILOT_MODEL", "gemini-1.5-pro"),
            enable_langsmith=bool(os.getenv("LANGSMITH_API_KEY")),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "clinical-ai-copilot"),
            mock_llm=os.getenv("COPILOT_MOCK_LLM", "0") == "1" or not google_api_key,
            enable_pubmed=os.getenv("COPILOT_ENABLE_PUBMED", "1") == "1",
            pubmed_max_results=int(os.getenv("COPILOT_PUBMED_MAX_RESULTS", "5")),
            enable_human_interrupt=os.getenv("COPILOT_ENABLE_HUMAN_INTERRUPT", "0") == "1",
        )
