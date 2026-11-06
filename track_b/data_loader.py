"""Track B 시계열 데이터 로더 및 누수 방지 전처리 모듈.

체크포인트 준수:
1. 시간순 분할: Train (70%) -> Validation (10%) -> Test (20%) (shuffle=False)
2. Scaler fit 제한: Scaler.fit()은 Train 구간에만 적용하고, Validation/Test는 transform()만 적용
3. 윈도우 생성 규칙: Look-ahead bias 방지 (t 시점 예측 시 t-1까지의 과거 값만 입력)
4. 모든 문서/코드 용어: Train, Validation, Test 통일
"""

from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


class TimeSeriesDataLoader:
    """시계열 주가 데이터 로드, 시간순 분할 및 Data Leakage 방지 전처리 클래스."""

    def __init__(
        self,
        ticker: str = "AAPL",
        start_date: str = "2021-01-01",
        end_date: str = "2024-01-01",
        cache_path: str = "track_b/data/AAPL_3y_daily.csv",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        window_size: int = 20,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율 합은 1.0이어야 합니다."
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.cache_path = cache_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.window_size = window_size

        self.df: Optional[pd.DataFrame] = None
        self.scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        self.is_scaler_fitted: bool = False

    def load_raw_data(self) -> pd.DataFrame:
        """데이터 로드: yfinance 온라인 시도 후 실패 시 로컬 캐시 활용 (Graceful Fallback)."""
        df = None
        # 1. 온라인 다운로드 시도
        try:
            downloaded = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
            )
            if not downloaded.empty and len(downloaded) >= 200:
                df = downloaded
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
        except Exception:
            df = None

        # 2. 로컬 캐시 Fallback
        if df is None or df.empty:
            if os.path.exists(self.cache_path):
                df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            else:
                raise FileNotFoundError(
                    f"yfinance 다운로드 실패 및 로컬 캐시({self.cache_path})를 찾을 수 없습니다."
                )

        # 결측치 정제 및 정렬
        df = df.dropna().sort_index()
        self.df = df
        return df

    def split_data_chronological(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """시간순 순차 분할 (Train -> Validation -> Test). shuffle=False."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_df = df.iloc[:train_end].copy()
        validation_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, validation_df, test_df

    def create_sliding_windows(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """과거 window_size개 시점 값(X)으로 다음 1시점 값(y) 예측하는 윈도우 생성.

        Look-ahead Bias를 원천 차단: y_t는 X_t의 마지막 시점 다음 시점 값.
        """
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i : i + self.window_size])
            y.append(series[i + self.window_size])
        return np.array(X), np.array(y)

    def prepare_datasets(self) -> Dict[str, Any]:
        """전체 데이터 파이프라인 실행: 로드 -> 분할 -> Fit(Train만) -> Transform -> 윈도우 생성."""
        raw_df = self.load_raw_data()
        close_prices = raw_df[["Close"]].values.astype(np.float32)

        # 1. 시간순 분할
        n = len(close_prices)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_raw = close_prices[:train_end]
        validation_raw = close_prices[train_end:val_end]
        test_raw = close_prices[val_end:]

        # 2. Scaler Fit: 오직 Train 구간만 fit (Data Leakage 방지 핵심 규칙)
        self.scaler.fit(train_raw)
        self.is_scaler_fitted = True

        # 3. Transform: Validation과 Test는 Train 기준 scaler로 변환만 수행
        train_scaled = self.scaler.transform(train_raw).flatten()

        # 윈도우 경계 보존: Validation과 Test의 초기 윈도우를 구성하기 위해 이전 구간의 끝부분 window_size일 연결
        val_series = np.concatenate([train_raw[-self.window_size:], validation_raw], axis=0)
        val_scaled = self.scaler.transform(val_series).flatten()

        test_series = np.concatenate([validation_raw[-self.window_size:], test_raw], axis=0)
        test_scaled = self.scaler.transform(test_series).flatten()

        # 4. 슬라이딩 윈도우 생성
        X_train, y_train = self.create_sliding_windows(train_scaled)
        X_val, y_val = self.create_sliding_windows(val_scaled)
        X_test, y_test = self.create_sliding_windows(test_scaled)

        # 3차원 텐서 형태 (Batch, Seq_Len, Feature_Dim)
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # 실제 원래 스케일 가격(Ground Truth for Evaluation)
        y_test_original = test_raw.flatten()
        assert len(y_test) == len(y_test_original), (
            f"Test 윈도우 타겟 개수({len(y_test)})와 실제 Test 데이터 개수({len(y_test_original)})가 일치해야 합니다."
        )

        return {
            "raw_df": raw_df,
            "train_raw": train_raw,
            "validation_raw": validation_raw,
            "test_raw": test_raw,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "y_test_original": y_test_original,
            "scaler": self.scaler,
            "split_indices": {
                "train_end": train_end,
                "val_end": val_end,
                "total": n,
            },
        }

    def inverse_transform(self, scaled_array: np.ndarray) -> np.ndarray:
        """스케일링된 1D/2D 배열을 원래 주가 스케일(달러)로 복원."""
        assert self.is_scaler_fitted, "Scaler가 먼저 Fit되어야 합니다."
        arr_2d = np.asarray(scaled_array, dtype=np.float32).reshape(-1, 1)
        return self.scaler.inverse_transform(arr_2d).flatten()
