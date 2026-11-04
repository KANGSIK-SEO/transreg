# 성능 진단 리포트 (Diagnosis Report)

## 1. 편향/분산(Bias/Variance) 진단 이론적 프레임워크

학습 데이터와 검증 데이터의 손실(Loss) 격차(Gap)를 분석하여 모델의 학습 상태를 체계적으로 진단합니다.

| 상태 | Train Loss | Validation Loss | Gap (Val - Train) | 진단 해석 | 해결 전략 |
|---|---|---|---|---|---|
| **High Bias (과소적합)** | 높음 | 높음 | 매우 작음 (비슷함) | 모델 복잡도 부족, 표현력 한계 | 모델 파라미터 증대, 레이어 동결 해제, 특성 추가 |
| **High Variance (과적합)** | 낮음 | 높음 | 매우 큼 (격차 심화) | 학습 데이터에 과적합, 일반화 실패 | 규제(Dropout, L2 Weight Decay), 데이터 증강, 조기 종료 |
| **Optimal (최적 상태)** | 낮음 | 낮음 | 안정적 유지 | 훈련 및 일반화 성능 동시 확보 | 현재 하이퍼파라미터 및 학습 스케줄 유지 |

---

## 2. Track A (Few-shot 전이학습) 편향/분산 진단

### 실험 결과 수치 요약
- **Linear Probing**:
  - 최종 Train Loss: `0.7327` | 최종 Validation Loss: `0.7369` | Loss Gap: `0.0042`
  - Test Accuracy: `70.40%` | 학습 파라미터 비율: `0.02%`
  - **진단**: `Good Fit`
  - **원인 분석**: Backbone(ResNet-18)을 완전히 동결(Freeze)하고 최종 분류기 레이어(2,565개 파라미터)만 학습시킴에 따라, 다운스트림 도메인 특화 특징 학습이 제한되어 Train Loss와 Validation Loss가 모두 일정 수준 이상으로 유지되는 편향이 관찰되었습니다.
  - **해결책**: 백본의 상위 합성곱 블록(Layer4) 또는 전체 레이어의 동결을 해제하여 모델 표현력을 확장합니다.

- **Full Fine-tuning**:
  - 최종 Train Loss: `0.1041` | 최종 Validation Loss: `0.6166` | Loss Gap: `0.5125`
  - Test Accuracy: `82.40%` | 학습 파라미터 비율: `100.0%`
  - **진단**: `High Variance 경향 (과적합)`
  - **원인 분석**: 전체 1,117만 개 파라미터의 동결을 해제하여 표현력을 대폭 끌어올렸으며, 적절한 규제(Weight Decay=1e-4)와 조기 종료를 통해 Linear Probing 대비 Test Accuracy가 대폭 향상되었습니다.
  - **해결책**: 데이터 부족(Few-shot) 환경에서 과적합을 방지하기 위해 RandomHorizontalFlip, RandomRotation 등 데이터 증강 기법을 지속 적용합니다.

---

## 3. Track B (시계열 주가 예측) 편향/분산 진단

### 최고 성능 모델 (GRU) 수치 요약
- 최종 Train Loss (MSE): `0.004648`
- 최종 Validation Loss (MSE): `0.001606`
- Loss Gap (Validation - Train): `0.003041`
- **진단**: `Balanced (Low Bias & Low Variance)`
- **원인 분석**:
  - Train Loss가 에포크 진행에 따라 점진적으로 수렴하면서도, Validation Loss와의 격차가 `0.003041`로 매우 안정적인 범위를 유지하고 있습니다.
  - LSTM 계층 간 Dropout(0.2)과 Adam 옵티마이저의 Weight Decay(1e-4)가 과도한 암기를 억제하고 시계열 일반화 성능을 성공적으로 확보했습니다.
- **향후 해결 및 고도화 전략**:
  - 시장 변동성이 급변하는 Out-of-Distribution 구간에 대비하여 Attention 메커니즘을 결합하거나, 멀티스케일 시계열 특성을 반영하는 구조로 고도화합니다.

---

## 4. 학습 곡선 시각화 증거

필수 시각화 1번 그래프 (`reports/figures/train_val_loss_curves.png`)에서 에포크별 Loss 수렴 궤적과 Validation 곡선의 동기화를 정밀하게 확인할 수 있습니다.
