from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .metrics import aurc, risk_coverage_curve
except ImportError:
    from metrics import aurc, risk_coverage_curve


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_risk_coverage(df: pd.DataFrame):
    set_plot_style()
    confidences = df["confidence_pairwise_dice"].to_numpy()
    risks = df["risk"].to_numpy()
    coverages, selected_risks = risk_coverage_curve(confidences, risks)
    score = aurc(confidences, risks)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(coverages, selected_risks, color="#2374ab", linewidth=2)
    ax.fill_between(coverages, 0, selected_risks, color="#2374ab", alpha=0.18)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selected risk")
    ax.set_title(f"Risk-coverage curve, AURC={score:.3f}")
    return fig, ax


def plot_cohort_reliability(summary_df: pd.DataFrame):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_df = summary_df[summary_df["cohort"].ne("All cohorts")].sort_values("aurc")
    sns.barplot(data=plot_df, x="aurc", y="cohort", color="#439a86", ax=ax)
    ax.set_xlabel("AURC, lower is better")
    ax.set_ylabel("")
    ax.set_title("Failure-detection reliability by cohort")
    return fig, ax
