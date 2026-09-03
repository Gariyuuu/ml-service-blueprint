"""Training: deterministic split, preprocessing, fit, evaluate, package, register."""

from mlservice.training.data import Split, load_dataset, split_dataset
from mlservice.training.evaluate import choose_threshold, evaluate_scores, score_distribution
from mlservice.training.model_card import render_model_card
from mlservice.training.model_factory import available_estimators, build_estimator
from mlservice.training.pipeline import TrainingResult, run_training
from mlservice.training.preprocessing import build_preprocessor

__all__ = [
    "Split",
    "TrainingResult",
    "available_estimators",
    "build_estimator",
    "build_preprocessor",
    "choose_threshold",
    "evaluate_scores",
    "load_dataset",
    "render_model_card",
    "run_training",
    "score_distribution",
    "split_dataset",
]
