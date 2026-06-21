from .metrics import (
    compute_depth_metrics,
    empirical_coverage,
    mean_interval_width,
    coverage_by_depth_bin,
    width_coverage_calibration_curve,
    three_group_accuracy,
    concordance_correlation_coefficient,
    confusion_matrix_3group,
    full_evaluation_report,
)
from .visualize import (
    plot_depth_panel,
    plot_coverage_calibration,
    plot_sequential_drift,
    plot_sizing_scatter,
    plot_confusion_matrix,
)

__all__ = [
    "compute_depth_metrics", "empirical_coverage", "mean_interval_width",
    "coverage_by_depth_bin", "width_coverage_calibration_curve",
    "three_group_accuracy", "concordance_correlation_coefficient",
    "confusion_matrix_3group", "full_evaluation_report",
    "plot_depth_panel", "plot_coverage_calibration", "plot_sequential_drift",
    "plot_sizing_scatter", "plot_confusion_matrix",
]
