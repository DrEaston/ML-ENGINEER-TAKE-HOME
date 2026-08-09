"""Reusable production scoring and evaluation interfaces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
)


@dataclass(frozen=True)
class ScoreResult:
    """Typed output for one scored transaction."""

    anomaly_score: float
    model_version: str


class FraudAnomalyScorer:
    """Package the fitted fraud-scoring components used in production.

    ``train_anomaly_scorer`` creates this object after fitting the preprocessing
    and anomaly detector on development data. The scorer keeps that fitted
    pipeline together with the required feature schema and model version, so
    production applies exactly the same preprocessing and model without
    retraining either component.

    Use ``save`` to serialize the packaged state, ``load`` to restore it in a
    production process, and ``score_one`` or ``score_batch`` to produce
    continuous anomaly scores for downstream ranking.
    """

    def __init__(
        self,
        pipeline,
        model_version: str,
        feature_columns=FEATURE_COLUMNS,
    ) -> None:
        # Keep every dependency needed for inference in one object.
        self.pipeline = pipeline
        self.model_version = model_version
        self.feature_columns = tuple(feature_columns)

    def score_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Score transactions with higher values consistently meaning riskier."""

        # Apply the same validation and fitted pipeline to the entire batch.
        self._validate(transactions)
        features = transactions.loc[:, self.feature_columns]

        # Both estimators use lower scores for more unusual observations.
        scores = -self.pipeline.score_samples(features)
        return pd.DataFrame(
            {
                "anomaly_score": scores,
                "model_version": self.model_version,
            },
            index=transactions.index,
        )

    def score_one(self, transaction: dict | pd.Series) -> ScoreResult:
        """Score one transaction using the same batch contract."""

        # Reuse batch scoring so single and batch results cannot diverge.
        row = self.score_batch(pd.DataFrame([dict(transaction)])).iloc[0]
        return ScoreResult(
            anomaly_score=float(row["anomaly_score"]),
            model_version=str(row["model_version"]),
        )

    def to_bundle(self) -> dict:
        """Return the portable state required for inference."""

        # Serialize the fitted pipeline together with its scoring contract.
        return {
            "pipeline": self.pipeline,
            "model_version": self.model_version,
            "feature_columns": self.feature_columns,
        }

    def save(self, path: str) -> None:
        """Serialize the portable scoring bundle."""

        # Persist only the fields required to recreate the inference interface.
        joblib.dump(self.to_bundle(), path)

    @classmethod
    def load(cls, path: str) -> "FraudAnomalyScorer":
        """Load a scoring bundle without any training dependencies."""

        # Reconstruct the scorer directly from the approved artifact.
        return cls(**joblib.load(path))

    def _validate(self, frame: pd.DataFrame) -> None:
        # Reject schema and value errors before they reach the model.
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame")
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        values = frame.loc[:, self.feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Features must contain only finite numeric values")


def train_anomaly_scorer(
    transactions: pd.DataFrame,
    *,
    model_name: str = "isolation_forest",
    random_state: int = 42,
    model_params: Mapping[str, Any] | None = None,
) -> FraudAnomalyScorer:
    """Create a fitted ``FraudAnomalyScorer`` during model development.

    This function selects the requested anomaly detector, builds a pipeline
    containing ``RobustScaler`` and that detector, and fits the pipeline on the
    supplied unlabeled transactions. It then returns a
    ``FraudAnomalyScorer`` that packages the fitted pipeline, expected feature
    schema, and model version for validation, serialization, and production
    inference.

    The function performs training only. Analyst review capacity is applied
    later when anomaly scores are ranked, and fraud labels are used separately
    by the evaluation workflow.
    """

    # Build candidates behind the same interface used by production.
    params = dict(model_params or {})
    if model_name == "isolation_forest":
        defaults = {
            "n_estimators": 200,
            "contamination": "auto",
            "random_state": random_state,
        }
        defaults.update(params)
        estimator = IsolationForest(**defaults)
    elif model_name == "local_outlier_factor":
        defaults = {
            "n_neighbors": 20,
            "contamination": "auto",
        }
        defaults.update(params)
        # Novelty mode is required to score transactions not seen during fit.
        defaults["novelty"] = True
        estimator = LocalOutlierFactor(**defaults)
    else:
        supported = "isolation_forest, local_outlier_factor"
        raise ValueError(f"Unsupported model {model_name!r}. Choose from: {supported}")

    features = transactions.loc[:, FEATURE_COLUMNS]

    pipeline = Pipeline(
        [("scaler", RobustScaler()), ("model", estimator)]
    )
    pipeline.fit(features)

    return FraudAnomalyScorer(
        pipeline=pipeline,
        model_version=f"fraud-{model_name}-v1",
    )


def evaluate_at_review_rate(frame: pd.DataFrame, review_rate: float) -> dict:
    """Evaluate a ranked anomaly score at a fixed analyst review budget."""

    # Measure fraud capture after selecting only the reviewable fraction.
    if not 0 < review_rate < 1:
        raise ValueError("review_rate must be between 0 and 1")
    required = {"Class", "anomaly_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing columns: {sorted(missing)}")

    threshold = frame["anomaly_score"].quantile(1 - review_rate)
    flagged = frame["anomaly_score"] >= threshold
    true_fraud = frame["Class"] == 1
    true_positives = int((flagged & true_fraud).sum())
    false_positives = int((flagged & ~true_fraud).sum())
    false_negatives = int((~flagged & true_fraud).sum())
    reviewed = int(flagged.sum())
    fraud_count = int(true_fraud.sum())
    return {
        "review_rate": review_rate,
        "reviewed_transactions": reviewed,
        "captured_fraud": true_positives,
        "missed_fraud": false_negatives,
        "precision": true_positives / reviewed if reviewed else 0.0,
        "recall": true_positives / fraud_count if fraud_count else 0.0,
        "false_positives_per_true_positive": (
            false_positives / true_positives if true_positives else np.inf
        ),
    }
