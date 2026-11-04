"""공통 평가 지표 모듈 (Common Evaluation Metrics).

모든 지표 계산에서 용어는 Train, Validation, Test를 엄격히 준수합니다.
"""

from typing import Dict
import numpy as np


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE) 계산."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE) 계산."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (MAPE, %) 계산."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    denominator = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """회귀 평가 지표 일괄 계산 (MAE, RMSE, MAPE)."""
    return {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


def calculate_relative_improvement(baseline_score: float, model_score: float) -> float:
    """베이스라인 대비 성능 개선율(%) 계산 (오차 지표 기준, 낮을수록 우수).

    Improvement (%) = ((Baseline - Model) / Baseline) * 100
    """
    if baseline_score == 0:
        return 0.0
    return float(((baseline_score - model_score) / baseline_score) * 100.0)
