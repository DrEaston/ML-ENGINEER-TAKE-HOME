"""Production-style anomaly scoring interface for credit-card transactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
)


@dataclass(frozen=True)
class ScoreResult:
    """Single-transaction score output."""

    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model_version: str


class FraudAnomalyScorer:
    """Bundle preprocessing, model, threshold, and input contract.

    `IsolationForest.score_samples` returns higher values for more normal
    observations, so this wrapper exposes `anomaly_score = -score_samples`.
    Higher score means more anomalous.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        threshold: float,
        feature_columns: Iterable[str] = FEATURE_COLUMNS,
        model_version: str = "fraud-anomaly-iforest-v1",
    ) -> None:
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.feature_columns = tuple(feature_columns)
        self.model_version = model_version

    @classmethod
    def fit(
        cls,
        transactions: pd.DataFrame,
        review_rate: float = 0.05,
        random_state: int = 42,
        n_estimators: int = 200,
        max_samples: str | int | float = "auto",
        model_version: str = "fraud-anomaly-iforest-v1",
    ) -> "FraudAnomalyScorer":
        """Fit an unsupervised anomaly model and choose a review-rate threshold.

        Labels must not be included in `transactions`; if a `Class` column is
        present it is ignored before training.
        """

        if not 0 < review_rate < 1:
            raise ValueError("review_rate must be between 0 and 1.")

        feature_frame = transactions.drop(columns=["Class"], errors="ignore")
        cls._validate_frame(feature_frame, FEATURE_COLUMNS)

        pipeline = Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        contamination=review_rate,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(feature_frame.loc[:, FEATURE_COLUMNS])
        train_scores = -pipeline.score_samples(feature_frame.loc[:, FEATURE_COLUMNS])
        threshold = float(np.quantile(train_scores, 1 - review_rate))
        return cls(pipeline=pipeline, threshold=threshold, model_version=model_version)

    def score_one(self, transaction: dict | pd.Series) -> ScoreResult:
        """Score a single transaction and return a typed result."""

        frame = pd.DataFrame([dict(transaction)])
        scored = self.score_batch(frame)
        row = scored.iloc[0]
        return ScoreResult(
            anomaly_score=float(row["anomaly_score"]),
            is_anomaly=bool(row["is_anomaly"]),
            threshold=float(row["threshold"]),
            model_version=str(row["model_version"]),
        )

    def score_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of transactions.

        Output contract:
        - `anomaly_score`: higher means more unusual.
        - `is_anomaly`: whether score is at or above the configured threshold.
        - `threshold`: threshold used for this model version.
        - `model_version`: bundled model identifier.
        """

        self._validate_frame(transactions, self.feature_columns)
        features = transactions.loc[:, self.feature_columns]
        anomaly_scores = -self.pipeline.score_samples(features)
        return pd.DataFrame(
            {
                "anomaly_score": anomaly_scores,
                "is_anomaly": anomaly_scores >= self.threshold,
                "threshold": self.threshold,
                "model_version": self.model_version,
            },
            index=transactions.index,
        )

    def save(self, path: str | Path) -> None:
        """Serialize the complete scoring bundle."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": self.threshold,
                "feature_columns": self.feature_columns,
                "model_version": self.model_version,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FraudAnomalyScorer":
        """Load a serialized scoring bundle."""

        bundle = joblib.load(path)
        return cls(
            pipeline=bundle["pipeline"],
            threshold=bundle["threshold"],
            feature_columns=bundle["feature_columns"],
            model_version=bundle["model_version"],
        )

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, feature_columns: Iterable[str]) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")

        feature_columns = tuple(feature_columns)
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        features = frame.loc[:, feature_columns]
        non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(features[column])]
        if non_numeric:
            raise TypeError(f"Feature columns must be numeric: {non_numeric}")

        values = features.to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError("Input contains NaN values; score only validated transactions.")
        if np.isinf(values).any():
            raise ValueError("Input contains infinite values; score only finite transactions.")

