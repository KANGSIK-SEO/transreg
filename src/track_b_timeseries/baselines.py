"""Track B 시계열 베이스라인 모델 모듈.

공정 비교 원칙:
- 모든 모델은 동일한 Test 구간에서 동일한 조건으로 평가됩니다.
- 딥러닝 모델 학습 전에 먼저 구현되어 기준 성능(Baseline)을 제공합니다.
"""

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from src.common.metrics import evaluate_regression_metrics


class NaiveForecastBaseline:
    """Naive Forecast: 바로 직전 시점 값을 그대로 예측. ŷ(t) = y(t-1)."""

    def __init__(self):
        self.name = "Naive Forecast"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X shape: (N, window_size, 1) 또는 (N, window_size).

        입력 윈도우의 가장 최근 값(마지막 인덱스)을 예측값으로 반환.
        """
        if X.ndim == 3:
            return X[:, -1, 0].copy()
        elif X.ndim == 2:
            return X[:, -1].copy()
        raise ValueError(f"예상치 못한 X 차원: {X.ndim}")


class SMABaseline:
    """단순 이동평균 (Simple Moving Average, SMA).

    ŷ(t) = (1/n) * Σ y(t-i), window n in {5, 10, 20}.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"SMA({window})"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """입력 윈도우의 최근 window개 값의 산술 평균."""
        if X.ndim == 3:
            seq = X[:, -self.window :, 0]
        else:
            seq = X[:, -self.window :]
        return np.mean(seq, axis=1)


class EMABaseline:
    """지수 이동평균 (Exponential Moving Average, EMA).

    ŷ(t) = α * y(t-1) + (1-α) * ŷ(t-1), α in {0.1, 0.3, 0.5}.
    """

    def __init__(self, alpha: float = 0.3):
        assert 0.0 < alpha <= 1.0, "alpha는 (0, 1] 범위여야 합니다."
        self.alpha = alpha
        self.name = f"EMA(alpha={alpha})"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """각 샘플의 윈도우 시퀀스에 대해 지수 평활을 적용하고 최종 다음 값을 예측."""
        if X.ndim == 3:
            seq = X[:, :, 0]
        else:
            seq = X

        predictions = []
        for row in seq:
            ema = row[0]
            for val in row[1:]:
                ema = self.alpha * val + (1.0 - self.alpha) * ema
            predictions.append(ema)
        return np.array(predictions, dtype=np.float32)


class LinearRegressionBaseline:
    """지연값(Lag Feature) 기반 선형 회귀 베이스라인."""

    def __init__(self):
        self.name = "Linear Regression (Lag)"
        self.model = LinearRegression()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train 데이터의 지연 특성으로 회귀 가중치 학습."""
        if X_train.ndim == 3:
            X_flat = X_train[:, :, 0]
        else:
            X_flat = X_train
        self.model.fit(X_flat, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X_flat = X[:, :, 0]
        else:
            X_flat = X
        return self.model.predict(X_flat)


def evaluate_all_baselines(
    data_dict: Dict[str, Any], inverse_fn
) -> Dict[str, Dict[str, Any]]:
    """모든 베이스라인 모델을 동일한 Test 구간에서 평가하고 메트릭을 반환합니다.

    평가는 역정규화된 원래 달러 가격 단위(y_test_original) 기준으로 수행합니다.
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test_orig = data_dict["y_test_original"]

    baselines = [
        NaiveForecastBaseline(),
        SMABaseline(window=5),
        SMABaseline(window=10),
        SMABaseline(window=20),
        EMABaseline(alpha=0.1),
        EMABaseline(alpha=0.3),
        EMABaseline(alpha=0.5),
    ]

    # 선형 회귀는 Train 구간으로 fit
    lr_model = LinearRegressionBaseline()
    lr_model.fit(X_train, y_train)
    baselines.append(lr_model)

    results = {}
    for model in baselines:
        scaled_preds = model.predict(X_test)
        orig_preds = inverse_fn(scaled_preds)
        metrics = evaluate_regression_metrics(y_test_orig, orig_preds)
        results[model.name] = {
            "metrics": metrics,
            "predictions_original": orig_preds,
            "predictions_scaled": scaled_preds,
        }

    return results
