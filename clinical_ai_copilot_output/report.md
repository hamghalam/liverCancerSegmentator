# Clinical AI Copilot Report

Case ID: demo_crlm_001

## Image Analysis Agent
- Tumor volume: None mL
- Lesion burden: unknown
- Ensemble confidence: None
- Segmentation uncertainty: unknown

## Radiomics Agent
- Extraction status: placeholder_ready_for_pyradiomics
- Response signal: unknown
- Feature families: shape_volume, shape_surface, first_order_intensity, texture_glcm_glrlm

## Medical Literature Agent
- CRLM multidisciplinary decision factors: Colorectal liver metastasis management should be discussed in a multidisciplinary tumor board.
- CRLM multidisciplinary decision factors: Resectability assessment depends on lesion distribution, future liver remnant, vascular involvement, extrahepatic disease, and patient fitness.
- CRLM multidisciplinary decision factors: Systemic therapy response and repeat imaging are commonly used to guide treatment sequencing.

## Clinical Reasoning Agent
- Treatment response likelihood: moderate
- Surgery suitability: requires_multidisciplinary_review
- Summary: Clinical synthesis: surgical suitability is uncertain and depends on lesion distribution, liver reserve, performance status, and multidisciplinary review.

## Evidence Verification Agent
- Status: needs_human_review
- Hallucination warning: True
- Unsupported claims: ['moderate', 'requires_multidisciplinary_review']

## Human-in-the-loop Review
- Required: True
- Review status: pending
- Review note: Human review interrupt disabled. Report is marked pending review.

## Safety Notice
This report is a research copilot output. It is not a diagnosis, treatment recommendation,
or replacement for oncologist/radiologist review.
