# 데이터 누수(Data Leakage) 방지 근거 리포트


## 1. 개요 및 원칙

시계열 예측에서 미래의 정보가 모델 학습 또는 전처리 과정에 유입되는 **Look-ahead Bias(미래 참조 편향)** 및 **Data Leakage(데이터 누수)**는 모델의 백테스트 성능을 비현실적으로 왜곡시키는 가장 치명적인 오류입니다.  
본 프로젝트는 다음 4대 체크포인트를 엄격히 준수하여 완벽한 누수 방지를 검증하고 실현했습니다.

---

## 2. 4대 누수 방지 체크포인트 준수 근거

| No | 체크 항목 | 준수 여부 | 실제 구현 및 기술적 증거 |
|---|---|:---:|---|
| **1** | **시간순 순차 분할 (Chronological Split)** | **준수 (PASS)** | 데이터를 무작위로 섞는(shuffle) 행위를 절대 금지하고, 시간의 흐름에 따라 정확히 `Train (70%) -> Validation (10%) -> Test (20%)`로 순차 분할하였습니다. `train_df.index.max() < val_df.index.min() < test_df.index.min()` 조건이 단위 테스트(`tests/test_data_leakage.py`)를 통해 자동 검증됩니다. |
| **2** | **전처리 Fit 구간 제한 (Train-only Fit)** | **준수 (PASS)** | 정규화 Scaler의 통계량 계산(`scaler.fit()`)은 오직 **Train 구간의 데이터**로만 수행되었습니다. Validation 및 Test 구간은 학습된 Scaler 인스턴스의 `transform()` 함수만을 적용하여, 미래 시점의 최소값(min)과 최대값(max)이 사전에 모델에 노출되지 않도록 물리적으로 차단했습니다. |
| **3** | **윈도우 생성 규칙 (Look-ahead Bias 차단)** | **준수 (PASS)** | 슬라이딩 윈도우 생성 시 입력 특성 $X_t$는 과거 $w$일($t-w \sim t-1$) 동안의 가격 시퀀스로만 구성되며, 타겟 레이블 $y_t$는 윈도우 바로 다음 시점($t$)의 단일 종가로 생성됩니다. 입력 시퀀스 내에 $t$ 시점 이후의 값이 절대 포함되지 않음을 단위 테스트로 증명했습니다. |
| **4** | **피쳐 생성 시점 및 증강 주의** | **준수 (PASS)** | 금융 시계열의 고유한 통계적 특성(단위근, 변동성 군집 등)을 보존하기 위해 비현실적인 시간 왜곡(Time-warping)이나 노이즈 주입(Jittering)을 엄격히 배제했습니다. 모든 이동평균 및 지연 특성은 분할된 각 세그먼트의 과거 시점만을 참조하도록 설계되었습니다. |

---

## 3. 코드 레벨 누수 방지 아키텍처

```python
# src/track_b_timeseries/data_loader.py 발췌

# 1. 시간순 분할 (shuffle=False)
train_raw = close_prices[:train_end]
validation_raw = close_prices[train_end:val_end]
test_raw = close_prices[val_end:]

# 2. Scaler Fit은 오직 Train 구간만! (핵심 원칙)
self.scaler.fit(train_raw)

# 3. Validation과 Test는 Transform만 적용
train_scaled = self.scaler.transform(train_raw)
val_scaled = self.scaler.transform(val_series)
test_scaled = self.scaler.transform(test_series)
```

---

## 4. 자동화 테스트 증거

`tests/test_data_leakage.py`의 단위 테스트 3종:
- `test_chronological_split_integrity`: 시간순 인덱스 무결성 검증
- `test_scaler_fit_train_only`: Scaler 통계치의 Train 격리 검증
- `test_no_look_ahead_bias_in_windows`: 윈도우 미래 시점 미포함 검증
- **결과**: `3 passed in 1.72s (100% PASS)`
