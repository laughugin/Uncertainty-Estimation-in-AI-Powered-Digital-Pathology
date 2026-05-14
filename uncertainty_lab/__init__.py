"""Uncertainty Lab — local toolkit for binary image classification with uncertainty evaluation."""

__version__ = "0.1.0"

from uncertainty_lab.pipeline.run import run_benchmark, run_pipeline

__all__ = ["run_benchmark", "run_pipeline", "__version__"]
