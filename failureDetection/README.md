# Confidence-Based Failure Detection

This folder contains the public, cleaned version of the uncertainty and failure-detection analysis for **Liver Cancer Segmentator**.

The goal is to estimate when an automatic liver/tumor segmentation is likely to fail, so low-confidence predictions can be prioritized for human review.

## What This Analysis Measures

- **Segmentation quality:** Dice similarity coefficient (DSC)
- **Failure risk:** `1 - DSC`
- **Confidence:** mean pairwise Dice agreement between ensemble folds
- **Reliability ranking:** area under the risk-coverage curve (AURC)
- **Metadata effects:** tumor volume, slice thickness, and cohort-level reliability

Lower AURC indicates that the confidence score is better at ranking failed cases early.

## Files

```text
failureDetection/
├── fd.ipynb              # Original exploratory notebook, kept for provenance
├── fd_clean.ipynb        # GitHub-ready notebook
├── failure_detection.py  # Data loading and metric collection
├── metrics.py            # Dice, confidence, risk, AURC utilities
├── plots.py              # Publication-style plotting helpers
└── example_config.yaml   # Example local path configuration
```

## Expected Data Layout

The clean notebook expects NIfTI masks with matching file names:

```text
data/
├── labels/
│   └── case_001.nii.gz
└── predictions/
    ├── ensemble/
    │   └── case_001.nii.gz
    ├── fold_0/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    └── fold_4/
```

Medical imaging data and model outputs are intentionally not committed to this repository.

## Run

From the repository root:

```bash
jupyter notebook failureDetection/fd_clean.ipynb
```

Update paths in the first notebook cell or copy `example_config.yaml` into a private local config.

## CV Summary

This project demonstrates a reliability layer for nnUNet medical image segmentation: ensemble-disagreement confidence scoring, cohort-level failure analysis, and AURC-based evaluation for identifying low-confidence liver/tumor predictions before clinical review.
