"""Tests for common evaluation metrics."""

import numpy as np
import pytest
from src.common.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_regression_metrics,
    calculate_relative_improvement,
)


def test_calculate_mae():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # abs diff: 2, 2, 3 -> mean = 7/3 = 2.3333...
    assert pytest.approx(calculate_mae(y_true, y_pred), 1e-4) == 2.33333333


def test_calculate_rmse():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # squared diff: 4, 4, 9 -> mean = 17/3 -> sqrt = 2.380476...
    assert pytest.approx(calculate_rmse(y_true, y_pred), 1e-4) == np.sqrt(17.0 / 3.0)


def test_calculate_mape():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    # percentage diff: 10/100=0.10, 10/200=0.05 -> mean = 0.075 -> 7.5%
    assert pytest.approx(calculate_mape(y_true, y_pred), 1e-4) == 7.5


def test_calculate_relative_improvement():
    # baseline MAE = 10.0, model MAE = 8.0 -> 20% improvement
    assert pytest.approx(calculate_relative_improvement(10.0, 8.0), 1e-4) == 20.0
    # baseline MAE = 10.0, model MAE = 12.0 -> -20% (degraded)
    assert pytest.approx(calculate_relative_improvement(10.0, 12.0), 1e-4) == -20.0


def test_evaluate_regression_metrics():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0, 20.0])
    res = evaluate_regression_metrics(y_true, y_pred)
    assert res["MAE"] == 0.0
    assert res["RMSE"] == 0.0
    assert res["MAPE"] == 0.0
