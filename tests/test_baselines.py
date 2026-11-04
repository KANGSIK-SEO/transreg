"""Tests for Track B baseline models."""

import numpy as np
import pytest
from src.track_b_timeseries.baselines import (
    NaiveForecastBaseline,
    SMABaseline,
    EMABaseline,
    LinearRegressionBaseline,
)


def test_naive_forecast():
    model = NaiveForecastBaseline()
    # shape (2, 4, 1): sample 0 has [1, 2, 3, 4], sample 1 has [10, 20, 30, 40]
    X = np.array([
        [[1.0], [2.0], [3.0], [4.0]],
        [[10.0], [20.0], [30.0], [40.0]],
    ])
    preds = model.predict(X)
    assert np.allclose(preds, [4.0, 40.0])


def test_sma_baseline():
    model = SMABaseline(window=3)
    X = np.array([
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
    ])
    # window=3 takes [3, 4, 5] -> mean is 4.0
    preds = model.predict(X)
    assert pytest.approx(preds[0]) == 4.0


def test_ema_baseline():
    model = EMABaseline(alpha=0.5)
    # sequence: 10.0, 20.0
    # step 0: ema = 10.0
    # step 1: ema = 0.5 * 20.0 + 0.5 * 10.0 = 15.0
    X = np.array([
        [[10.0], [20.0]],
    ])
    preds = model.predict(X)
    assert pytest.approx(preds[0]) == 15.0


def test_linear_regression_baseline():
    model = LinearRegressionBaseline()
    # Simple linear pattern y = 2 * last_val
    X_train = np.array([
        [[1.0], [2.0]],
        [[2.0], [3.0]],
        [[3.0], [4.0]],
        [[4.0], [5.0]],
    ])
    y_train = np.array([3.0, 4.0, 5.0, 6.0])
    model.fit(X_train, y_train)

    X_test = np.array([
        [[5.0], [6.0]],
    ])
    preds = model.predict(X_test)
    assert pytest.approx(preds[0], abs=1e-3) == 7.0
