"""Tests for Track B Data Leakage prevention rules.

체크 항목:
1. 시간순 분할: 반드시 시간 순서대로 Train -> Validation -> Test 분할 (shuffle 금지)
2. 전처리 Fit 구간 제한: Scaler.fit()은 Train 구간에만 적용
3. 윈도우 생성 규칙: Look-ahead bias 방지
"""

import numpy as np
import pytest
from track_b.data_loader import TimeSeriesDataLoader


@pytest.fixture
def data_loader():
    return TimeSeriesDataLoader(
        cache_path="track_b/data/AAPL_3y_daily.csv",
        window_size=20,
    )


def test_chronological_split_integrity(data_loader):
    raw_df = data_loader.load_raw_data()
    train_df, val_df, test_df = data_loader.split_data_chronological(raw_df)

    # 1. 길이 검증
    assert len(train_df) + len(val_df) + len(test_df) == len(raw_df)

    # 2. 시간 순서 검증 (Train 최신 날짜 < Validation 가장 오래된 날짜 < Test 가장 오래된 날짜)
    assert train_df.index.max() < val_df.index.min(), "Train과 Validation 순서가 뒤섞였습니다."
    assert val_df.index.max() < test_df.index.min(), "Validation과 Test 순서가 뒤섞였습니다."


def test_scaler_fit_train_only(data_loader):
    """Scaler fit()이 오직 Train 데이터 통계로만 결정되는지 검증."""
    datasets = data_loader.prepare_datasets()
    scaler = datasets["scaler"]
    train_raw = datasets["train_raw"]
    test_raw = datasets["test_raw"]

    # Scaler의 min, max가 오직 train_raw의 min, max와 일치해야 함
    expected_min = float(np.min(train_raw))
    expected_max = float(np.max(train_raw))

    assert pytest.approx(scaler.data_min_[0], 1e-5) == expected_min
    assert pytest.approx(scaler.data_max_[0], 1e-5) == expected_max

    # 만약 test_raw에 train_raw보다 크거나 작은 값이 있어도 scaler min/max에 반영되지 않아야 함
    # (즉, Test 데이터 정보 누수가 발생하지 않음)
    full_min = float(np.min(np.concatenate([train_raw, test_raw])))
    full_max = float(np.max(np.concatenate([train_raw, test_raw])))
    if full_min < expected_min or full_max > expected_max:
        assert scaler.data_min_[0] != full_min or scaler.data_max_[0] != full_max


def test_no_look_ahead_bias_in_windows(data_loader):
    """윈도우 생성 시 미래 시점 값이 과거 입력에 포함되지 않는지 검증."""
    series = np.arange(100, dtype=np.float32)
    X, y = data_loader.create_sliding_windows(series)

    for i in range(len(X)):
        # 입력 윈도우 X[i]의 마지막 원소 바로 다음 값이 y[i]여야 함
        assert X[i][-1] == y[i] - 1
        # y[i]는 X[i] 안에 절대 포함되어선 안 됨
        assert y[i] not in X[i]
