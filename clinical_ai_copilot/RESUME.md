# Resume Positioning

## Project title

Clinical AI Copilot for Liver Cancer Segmentation and Treatment Review

## One-line summary

Built a LangGraph-based multi-agent clinical AI copilot that combines nnUNet CT segmentation, uncertainty estimation, radiology text, patient context, guideline evidence, hallucination checks, and human-in-the-loop report generation for colorectal liver metastasis review.

## Resume bullets

* Designed a LangGraph multi-agent workflow with six specialized agents for image analysis, radiomics, medical literature retrieval, clinical reasoning, evidence verification, and report generation.
* Integrated nnUNet liver/tumor segmentation metadata, tumor volume, ensemble confidence, and uncertainty estimates into a downstream clinical reasoning pipeline.
* Built a RAG-ready evidence layer for PubMed and guideline retrieval, enabling traceable clinical summaries grounded in supporting evidence.
* Added LLM safety controls including unsupported-claim detection, hallucination warnings, explicit human-review gates, and explainable audit logs.
* Implemented Google Gemini integration with optional LangSmith tracing for observability, debugging, and agent evaluation.

## Keywords

LangGraph, Multi-Agent AI, Agent Orchestration, RAG, Tool Calling, Medical AI, LLM Evaluation, AI Safety, Human-in-the-loop, Explainability, nnUNet, Radiomics, Clinical Decision Support

## Interview explanation

This project starts from a real medical imaging model rather than a toy chatbot. The segmentation system produces tumor masks, tumor volume, and uncertainty from five-fold nnUNet ensemble inference. The Clinical AI Copilot then treats those outputs as tools inside a LangGraph workflow and combines them with radiology text, patient context, and guideline evidence. A separate verification agent checks whether the reasoning is supported by evidence and forces human review before any clinical interpretation is trusted.
