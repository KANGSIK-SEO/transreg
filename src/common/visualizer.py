"""필수 실험 결과 시각화 모듈 (4종 필수 시각화).

1. Train/Validation Loss 곡선 (Gap으로 편향/분산 진단)
2. 예측 vs 실제 그래프 (시계열: 실제값 파란선 + 예측값 빨간선)
3. 베이스라인 비교 막대그래프 (모델별 MAE/RMSE/MAPE 비교)
4. 오분류 이미지 갤러리 (태그별 그룹화된 오분류 이미지)
"""

from typing import Dict, List, Any
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 한글/영어 폰트 호환 설정 (AppleGothic 우선)
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False



def plot_train_val_loss_curves(
    track_a_history: Dict[str, List[float]],
    track_b_history: Dict[str, List[float]],
    save_path: str = "reports/figures/train_val_loss_curves.png",
) -> None:
    """1. Train/Validation Loss 곡선 (Gap으로 편향/분산 진단)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Track A
    epochs_a = range(1, len(track_a_history["train_loss"]) + 1)
    axes[0].plot(epochs_a, track_a_history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs_a, track_a_history["validation_loss"], "r--s", label="Validation Loss", linewidth=2)
    axes[0].set_title("Track A (Transfer Learning) Loss Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # Track B
    epochs_b = range(1, len(track_b_history["train_loss"]) + 1)
    axes[1].plot(epochs_b, track_b_history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[1].plot(epochs_b, track_b_history["validation_loss"], "r--s", label="Validation Loss", linewidth=2)
    axes[1].set_title("Track B (Time Series LSTM) Loss Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_timeseries_prediction_vs_actual(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    dates: List[Any],
    model_name: str = "LSTM",
    save_path: str = "track_b/reports/figures/timeseries_prediction_vs_actual.png",
) -> None:
    """2. 시계열 예측 vs 실제 그래프 (실제값 파란선 + 예측값 빨간선)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))

    x_axis = range(len(actual_prices))
    plt.plot(x_axis, actual_prices, color="blue", label="Actual Price ($)", linewidth=2.0)
    plt.plot(x_axis, predicted_prices, color="red", linestyle="--", label=f"Predicted Price ({model_name})", linewidth=2.0)

    plt.title(f"Track B: Time Series Stock Price Prediction vs Actual ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Test Timeline (Trading Days)", fontsize=12)
    plt.ylabel("Stock Price (USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_baseline_comparison_barplot(
    comparison_dict: Dict[str, Dict[str, float]],
    save_path: str = "track_b/reports/figures/baseline_comparison_barplot.png",
) -> None:
    """3. 베이스라인 vs 딥러닝 모델 비교 막대그래프 (MAE & RMSE)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_names = list(comparison_dict.keys())
    mae_values = [comparison_dict[m]["MAE"] for m in model_names]
    rmse_values = [comparison_dict[m]["RMSE"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, mae_values, width, label="MAE ($)", color="#4A90E2")
    bars2 = ax.bar(x + width / 2, rmse_values, width, label="RMSE ($)", color="#50E3C2")

    ax.set_title("Track B: Baseline vs Deep Learning Models Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model Name", fontsize=12)
    ax.set_ylabel("Error Metric (USD, Lower is Better)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    # 값 텍스트 표시
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_analysis_gallery(
    misclassified_items: List[Dict[str, Any]],
    num_samples: int = 12,
    save_path: str = "track_a/reports/figures/error_analysis_gallery.png",
) -> None:
    """4. 오분류 이미지 갤러리 (태그별 그룹화된 오분류 샘플)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    selected = misclassified_items[:num_samples]
    n = len(selected)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n:
            item = selected[i]
            img = item["raw_image"]
            axes[i].imshow(np.clip(img, 0, 1))
            tag_short = item["tag"].split(" ")[0]
            title_text = f"True: {item['true_class']}\nPred: {item['pred_class']} ({item['confidence']:.2f})\n[{tag_short}]"
            axes[i].set_title(title_text, fontsize=9, color="darkred", fontweight="bold")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("Track A: Misclassified Images Gallery by Failure Tag", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
