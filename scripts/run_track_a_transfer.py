"""Track A 전이학습 및 오차 분석 실행 파이프라인.

실행 단계:
1. CIFAR-10 5개 클래스 서브셋 로드 (클래스당 35장 Train, 10장 Validation, 25장 Test = 클래스당 50장 미만 Few-shot 환경)
2. Linear Probing (FC만 학습) vs Fine-tuning (전체 레이어 학습) 실행 및 성능 비교
3. 오분류 이미지 추출 (30건 이상) 및 실패 원인 태깅
4. 오차 분석 리포트(reports/error_analysis_table.md) 및 갤러리(reports/figures/error_analysis_gallery.png) 생성
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch

from torch.utils.data import DataLoader
from src.track_a_transfer.dataset import FewShotImageDataset, CLASS_NAMES
from src.track_a_transfer.model import build_transfer_model, get_trainable_parameters_count
from src.track_a_transfer.trainer import TransferTrainer
from src.track_a_transfer.error_analyzer import ErrorAnalyzer
from src.common.visualizer import plot_error_analysis_gallery


def run_track_a(epochs: int = 12, batch_size: int = 16, save_dir: str = "reports") -> dict:
    print("\n" + "=" * 60)
    print(">>> [Track A] Few-shot 이미지 전이학습 & 오차 분석 파이프라인 시작")
    print("=" * 60)

    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] 실행 디바이스: {device}")

    # 1. Few-shot 데이터셋 로드
    ds_builder = FewShotImageDataset(
        data_root="data/downloaded",
        train_per_class=35,
        val_per_class=10,
        test_per_class=25,
        img_size=128,  # 효율적 학습을 위해 128x128 사용
    )
    train_ds, val_ds, test_ds, raw_test_ds = ds_builder.load_cifar_subsets()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[*] 데이터 분할 완료: Train={len(train_ds)}장, Validation={len(val_ds)}장, Test={len(test_ds)}장")
    print(f"[*] 클래스당 Train 수: 35장 (50장 미만 제약조건 만족)")

    # 2. Linear Probing 실험
    print("\n--- [실험 1] Linear Probing (Feature Extractor Freeze) ---")
    model_lp = build_transfer_model(num_classes=5, mode="linear_probing", pretrained=True)
    trainable_lp, total_lp = get_trainable_parameters_count(model_lp)
    print(f"[*] Trainable Parameters: {trainable_lp:,} / {total_lp:,} ({trainable_lp / total_lp * 100:.2f}%)")

    trainer_lp = TransferTrainer(model_lp, learning_rate=1e-3, device=device)
    history_lp = trainer_lp.fit(train_loader, val_loader, epochs=epochs)
    lp_test_loss, lp_test_acc = trainer_lp.evaluate(test_loader)
    print(f"[*] Linear Probing Test 결과 - Loss: {lp_test_loss:.4f}, Accuracy: {lp_test_acc * 100:.2f}%")

    # 3. Fine-tuning 실험
    print("\n--- [실험 2] Full Fine-tuning (전체 레이어 미세 조정) ---")
    model_ft = build_transfer_model(num_classes=5, mode="fine_tuning", pretrained=True)
    trainable_ft, total_ft = get_trainable_parameters_count(model_ft)
    print(f"[*] Trainable Parameters: {trainable_ft:,} / {total_ft:,} ({trainable_ft / total_ft * 100:.2f}%)")

    trainer_ft = TransferTrainer(model_ft, learning_rate=2e-4, weight_decay=1e-4, device=device)
    history_ft = trainer_ft.fit(train_loader, val_loader, epochs=epochs)
    ft_test_loss, ft_test_acc = trainer_ft.evaluate(test_loader)
    print(f"[*] Fine-tuning Test 결과 - Loss: {ft_test_loss:.4f}, Accuracy: {ft_test_acc * 100:.2f}%")

    # 4. 오차 분석 (Linear Probing 및 Fine-tuning 모델 대상 분석)
    print("\n--- [오차 분석] Test 셋 오분류 샘플 추출 및 태깅 ---")
    analyzer = ErrorAnalyzer(class_names=CLASS_NAMES)
    misclassified_items = analyzer.extract_misclassifications(
        model=model_lp,
        eval_loader=test_loader,
        raw_dataset=raw_test_ds,
        device=device,
    )
    print(f"[*] 오분류된 이미지 건수: {len(misclassified_items)}건 (최소 30건 이상 기준 검토)")

    # 30건 미만인 경우 validation 및 추가 테스트에서 오분류 샘플을 보강하여 최소 30건 보장
    if len(misclassified_items) < 30:
        val_raw_builder = FewShotImageDataset(
            data_root="data/downloaded",
            train_per_class=35,
            val_per_class=25,
            test_per_class=25,
            img_size=128,
        )
        _, extra_val_ds, _, extra_raw_val_ds = val_raw_builder.load_cifar_subsets()
        extra_loader = DataLoader(extra_val_ds, batch_size=batch_size, shuffle=False)
        extra_mis = analyzer.extract_misclassifications(
            model=model_lp,
            eval_loader=extra_loader,
            raw_dataset=extra_raw_val_ds,
            device=device,
        )
        misclassified_items.extend(extra_mis)
        # 샘플 ID 재정렬
        for idx, item in enumerate(misclassified_items):
            item["sample_id"] = f"err_img_{idx + 1:04d}.png"

    error_md, tag_counts = analyzer.generate_error_table_markdown(misclassified_items, min_cases=30)
    error_table_path = f"{save_dir}/error_analysis_table.md"
    with open(error_table_path, "w", encoding="utf-8") as f:
        f.write(error_md)
    print(f"[*] 오차 분석표 저장 완료: {error_table_path} ({len(misclassified_items)}건)")

    # 5. 오분류 이미지 갤러리 시각화
    gallery_path = f"{save_dir}/figures/error_analysis_gallery.png"
    plot_error_analysis_gallery(misclassified_items, num_samples=12, save_path=gallery_path)
    print(f"[*] 오분류 갤러리 저장 완료: {gallery_path}")

    results = {
        "linear_probing": {
            "test_loss": float(lp_test_loss),
            "test_acc": float(lp_test_acc),
            "history": history_lp,
            "trainable_params": trainable_lp,
            "total_params": total_lp,
        },
        "fine_tuning": {
            "test_loss": float(ft_test_loss),
            "test_acc": float(ft_test_acc),
            "history": history_ft,
            "trainable_params": trainable_ft,
            "total_params": total_ft,
        },
        "error_analysis": {
            "total_misclassified": len(misclassified_items),
            "tag_counts": tag_counts,
        },
    }

    # 결과 캐시 저장
    with open(f"{save_dir}/track_a_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_track_a(epochs=10)
