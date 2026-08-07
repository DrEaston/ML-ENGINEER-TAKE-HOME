from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fraud_scorer import FEATURE_COLUMNS, FraudAnomalyScorer


def make_transactions(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_rows, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    frame["Time"] = np.arange(n_rows, dtype=float)
    frame["Amount"] = np.abs(frame["Amount"] * 100)
    return frame


def test_serialization_round_trip_preserves_scores(tmp_path: Path) -> None:
    train = make_transactions()
    scorer = FraudAnomalyScorer.fit(train, review_rate=0.05, n_estimators=25)
    before = scorer.score_batch(train.head(10))

    model_path = tmp_path / "fraud_scorer.joblib"
    scorer.save(model_path)
    loaded = FraudAnomalyScorer.load(model_path)
    after = loaded.score_batch(train.head(10))

    pd.testing.assert_frame_equal(before, after)


def test_score_rejects_missing_feature() -> None:
    train = make_transactions()
    scorer = FraudAnomalyScorer.fit(train, review_rate=0.05, n_estimators=25)
    bad = train.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        scorer.score_batch(bad)


def test_score_rejects_nan_and_infinite_values() -> None:
    train = make_transactions()
    scorer = FraudAnomalyScorer.fit(train, review_rate=0.05, n_estimators=25)

    with_nan = train.head(3).copy()
    with_nan.loc[0, "Amount"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        scorer.score_batch(with_nan)

    with_inf = train.head(3).copy()
    with_inf.loc[0, "Amount"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        scorer.score_batch(with_inf)

