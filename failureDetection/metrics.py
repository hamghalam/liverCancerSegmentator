from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def dice_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_value: int = 2,
    empty_score: float = 1.0,
) -> float:
    """Calculate Dice similarity coefficient for one segmentation label."""
    true_mask = y_true == label_value
    pred_mask = y_pred == label_value

    denominator = true_mask.sum() + pred_mask.sum()
    if denominator == 0:
        return empty_score

    intersection = np.logical_and(true_mask, pred_mask).sum()
    return float(2.0 * intersection / denominator)


def pairwise_dice_confidence(
    predictions: Iterable[np.ndarray],
    label_value: int = 2,
) -> float:
    """Use mean pairwise Dice across ensemble folds as a confidence score."""
    prediction_list = list(predictions)
    if len(prediction_list) < 2:
        raise ValueError("At least two predictions are required.")

    scores = [
        dice_coefficient(left, right, label_value=label_value)
        for left, right in combinations(prediction_list, 2)
    ]
    return float(np.nanmean(scores))


def tumor_volume_mm3(segmentation: np.ndarray, spacing: tuple[float, ...], label_value: int = 2) -> float:
    """Estimate tumor volume in cubic millimeters for a label mask."""
    voxel_volume = float(np.prod(spacing))
    return float(np.sum(segmentation == label_value) * voxel_volume)


def reliability_risk(dice_score: float) -> float:
    """Convert segmentation quality into failure risk."""
    return float(1.0 - dice_score)


def risk_coverage_curve(confidences: np.ndarray, risks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute selected risk as lower-confidence cases are rejected.

    Coverage is the fraction of retained cases. Cases are sorted from highest to
    lowest confidence, then the cumulative mean risk is measured.
    """
    confidences = np.asarray(confidences, dtype=float)
    risks = np.asarray(risks, dtype=float)

    if confidences.shape != risks.shape:
        raise ValueError("confidences and risks must have the same shape.")
    if confidences.ndim != 1:
        raise ValueError("confidences and risks must be one-dimensional.")
    if len(confidences) == 0:
        raise ValueError("At least one sample is required.")

    order = np.argsort(-confidences)
    sorted_risks = risks[order]
    retained = np.arange(1, len(sorted_risks) + 1)

    coverages = retained / len(sorted_risks)
    selected_risks = np.cumsum(sorted_risks) / retained
    return coverages, selected_risks


def aurc(confidences: np.ndarray, risks: np.ndarray) -> float:
    """Area under the risk-coverage curve."""
    coverages, selected_risks = risk_coverage_curve(confidences, risks)
    coverages = np.concatenate([[0.0], coverages])
    selected_risks = np.concatenate([[selected_risks[0]], selected_risks])
    return float(np.trapz(selected_risks, coverages))


def map_to_cohort(filename: str) -> str:
    """Map dataset-specific file names to readable cohort names."""
    if "010A_000" in filename:
        return "Resection"
    if "010C_000" in filename:
        return "Chemotherapy"
    if "010_002" in filename:
        return "All stages-MDA"
    if "mcrc" in filename.lower():
        return "TCIA"
    return "All stages-MSK"
