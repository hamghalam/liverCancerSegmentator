# Clinical AI Copilot

Clinical AI Copilot is a portfolio-ready multi-agent extension for the Liver Cancer Segmentator project. It uses LangGraph to coordinate image analysis, radiomics, medical literature retrieval, clinical reasoning, evidence verification, and final report generation for colorectal liver metastasis case review.

This module is designed for research and engineering demonstration. It does not provide autonomous diagnosis or treatment recommendations.

## Why this matters

The base repository already performs nnUNet liver/tumor segmentation and confidence-based failure detection. This folder adds the clinical AI orchestration layer that industrial teams often look for:

* LangGraph agent orchestration
* Multi-agent AI workflow design
* RAG-ready medical literature retrieval
* Tool calling over segmentation metadata
* LLM evaluation and hallucination checks
* Human-in-the-loop safety gates
* Explainable report generation

## Agents

| Agent | Role | Output |
| --- | --- | --- |
| Image Analysis Agent | Reads nnUNet confidence metadata, tumor volume, lesion burden, and uncertainty. | Structured image summary |
| Radiomics Agent | Prepares a PyRadiomics-ready feature summary and response signal placeholder. | Radiomics feature summary |
| Medical Literature Agent | Retrieves local guideline snippets and prepares PubMed/RAG queries. | Evidence list |
| Clinical Reasoning Agent | Synthesizes imaging, radiomics, report text, patient context, and evidence. | Cautious clinical assessment |
| Evidence Verification Agent | Checks unsupported claims and hallucination risk. | Safety verification |
| Report Generator | Produces a clinician-readable report with traceable inputs. | Markdown report |

## Install

From the repository root:

```bash
pip install -e ".[copilot]"
```

For offline demos without calling Gemini:

```bash
set COPILOT_MOCK_LLM=1
python -m clinical_ai_copilot.run_demo
```

For Google Gemini and LangSmith tracing:

```bash
set GOOGLE_API_KEY=your_google_api_key
set LANGSMITH_API_KEY=your_langsmith_api_key
set LANGSMITH_PROJECT=clinical-ai-copilot
python -m clinical_ai_copilot.run_demo
```

On Linux/macOS, use `export` instead of `set`.

## Use with nnUNet confidence metadata

First run the existing confidence pipeline:

```bash
python -m failureDetection.confidence_cli \
  --input data/input \
  --output data/output \
  --model models/nnUNetTrainerV2__nnUNetPlansv2.1 \
  --case-id patient_001
```

Then point a case JSON file at the generated metadata:

```json
{
  "case_id": "patient_001",
  "ct_path": "data/input/patient_001_0000.nii.gz",
  "segmentation_metadata_path": "data/output/patient_001_confidence.json",
  "radiology_report": "Multiple colorectal liver metastases...",
  "patient_profile": {
    "age": 62,
    "sex": "female",
    "primary_cancer": "colorectal cancer",
    "performance_status": "ECOG 1"
  },
  "clinical_question": "Assess treatment response likelihood and surgical review suitability.",
  "guidelines": [],
  "human_review_required": true,
  "audit_log": []
}
```

Run:

```bash
python -m clinical_ai_copilot.run_demo --case path/to/case.json --output reports/patient_001.md
```

## Architecture

```text
CT Scan + Radiology Report + Patient Data + Guidelines
        |
        v
Image Analysis Agent -> Radiomics Agent -> Medical Literature Agent
        |                    |                    |
        +--------------------+--------------------+
                             v
                  Clinical Reasoning Agent
                             |
                             v
                Evidence Verification Agent
                             |
                             v
                     Report Generator
```

## Resume bullets

* Built a LangGraph-based multi-agent Clinical AI Copilot for CT-driven colorectal liver metastasis assessment.
* Integrated nnUNet segmentation metadata, uncertainty scoring, radiology text, patient context, and guideline evidence into a structured agent workflow.
* Added evidence verification and human-in-the-loop safety gates to reduce hallucination risk in medical AI outputs.
* Designed a RAG-ready literature agent for PubMed/guideline retrieval and explainable clinical report generation.

## Safety note

This module is for software engineering, medical AI research, and portfolio demonstration. All outputs require expert clinical review.
