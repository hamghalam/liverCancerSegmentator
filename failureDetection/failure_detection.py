from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from .metrics import (
        aurc,
        dice_coefficient,
        map_to_cohort,
        pairwise_dice_confidence,
        reliability_risk,
        risk_coverage_curve,
        tumor_volume_mm3,
    )
except ImportError:
    from metrics import (
        aurc,
        dice_coefficient,
        map_to_cohort,
        pairwise_dice_confidence,
        reliability_risk,
        risk_coverage_curve,
        tumor_volume_mm3,
    )


def read_nifti(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Load a NIfTI segmentation and return array plus voxel spacing."""
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), tuple(float(v) for v in image.GetSpacing())


def collect_case_metrics(
    reference_dir: Path,
    ensemble_dir: Path,
    fold_dirs: list[Path],
    label_value: int = 2,
) -> pd.DataFrame:
    """
    Collect failure-detection metrics for one reference directory.

    Expected file layout:
    - reference_dir/case_id.nii.gz
    - ensemble_dir/case_id.nii.gz
    - fold_dirs[i]/case_id.nii.gz
    """
    rows = []
    reference_files = sorted(reference_dir.glob("*.nii.gz"))

    for reference_path in reference_files:
        case_id = reference_path.name
        ensemble_path = ensemble_dir / case_id
        fold_paths = [fold_dir / case_id for fold_dir in fold_dirs]

        if not ensemble_path.exists() or not all(path.exists() for path in fold_paths):
            rows.append(
                {
                    "case_id": case_id,
                    "cohort": map_to_cohort(case_id),
                    "status": "missing_prediction",
                }
            )
            continue

        reference, spacing = read_nifti(reference_path)
        ensemble_prediction, _ = read_nifti(ensemble_path)
        fold_predictions = [read_nifti(path)[0] for path in fold_paths]

        dice = dice_coefficient(reference, ensemble_prediction, label_value=label_value)
        confidence = pairwise_dice_confidence(fold_predictions, label_value=label_value)

        rows.append(
            {
                "case_id": case_id,
                "cohort": map_to_cohort(case_id),
                "status": "ok",
                "dice": dice,
                "risk": reliability_risk(dice),
                "confidence_pairwise_dice": confidence,
                "tumor_volume_mm3": tumor_volume_mm3(reference, spacing, label_value=label_value),
                "slice_thickness_mm": spacing[2] if len(spacing) >= 3 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize AURC and correlation by cohort and globally."""
    valid = df[df["status"].eq("ok")].copy()
    groups = [("All cohorts", valid), *valid.groupby("cohort")]
    rows = []

    for cohort, group in groups:
        if len(group) < 2:
            continue

        rows.append(
            {
                "cohort": cohort,
                "cases": len(group),
                "mean_dice": group["dice"].mean(),
                "mean_risk": group["risk"].mean(),
                "aurc": aurc(group["confidence_pairwise_dice"].to_numpy(), group["risk"].to_numpy()),
                "spearman_confidence_risk": group["confidence_pairwise_dice"].corr(
                    group["risk"], method="spearman"
                ),
                "pearson_confidence_risk": group["confidence_pairwise_dice"].corr(
                    group["risk"], method="pearson"
                ),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "aurc",
    "collect_case_metrics",
    "read_nifti",
    "risk_coverage_curve",
    "summarize_reliability",
]
