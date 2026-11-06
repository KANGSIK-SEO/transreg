"""Layer 1 & Layer 2 통합 검증: README와 실제 실험 결과(JSON)의 100% 일치 자동 감사 테스트."""

import os
import json
import re
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
README_PATH = os.path.join(REPO_ROOT, "README.md")
TRACK_A_JSON = os.path.join(REPO_ROOT, "track_a", "reports", "track_a_results.json")
TRACK_B_JSON = os.path.join(REPO_ROOT, "track_b", "reports", "track_b_results.json")
@pytest.fixture(scope="module")
def readme_content():
    assert os.path.exists(README_PATH), "README.md 파일이 존재하지 않습니다."
    with open(README_PATH, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def track_a_data():
    assert os.path.exists(TRACK_A_JSON), "track_a_results.json이 존재하지 않습니다."
    with open(TRACK_A_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def track_b_data():
    assert os.path.exists(TRACK_B_JSON), "track_b_results.json이 존재하지 않습니다."
    with open(TRACK_B_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_readme_track_a_metrics_sync(readme_content, track_a_data):
    """README에 기재된 Track A 메트릭(Accuracy, Loss, 파라미터 수)이 실제 JSON과 정확히 일치하는지 검증."""
    lp_acc = track_a_data["linear_probing"]["test_acc"] * 100
    ft_acc = track_a_data["fine_tuning"]["test_acc"] * 100
    total_params = track_a_data["linear_probing"]["total_params"]
    lp_trainable = track_a_data["linear_probing"]["trainable_params"]

    # README 내 정확도 문자열 표기 확인
    assert f"{lp_acc:.2f}%" in readme_content, f"README에 Linear Probing 정확도 {lp_acc:.2f}% 미일치"
    assert f"{ft_acc:.2f}%" in readme_content, f"README에 Fine-tuning 정확도 {ft_acc:.2f}% 미일치"

    # 파라미터 수 표기 일치 확인
    assert f"{lp_trainable:,}" in readme_content, f"README에 학습 가능 파라미터 수 {lp_trainable} 미일치"
    assert f"{total_params:,}" in readme_content, f"README에 전체 파라미터 수 {total_params} 미일치"


def test_readme_track_b_metrics_sync(readme_content, track_b_data):
    """README에 기재된 Track B 주가 예측 모델들의 MAE, RMSE가 실제 JSON과 정확히 일치하는지 검증."""
    comparison = track_b_data["comparison"]

    for model_name, metrics in comparison.items():
        mae_str = f"${metrics['MAE']:.2f}"
        rmse_str = f"${metrics['RMSE']:.2f}"
        assert mae_str in readme_content, f"README에 {model_name}의 MAE ({mae_str}) 표기 누락 또는 불일치"
        assert rmse_str in readme_content, f"README에 {model_name}의 RMSE ({rmse_str}) 표기 누락 또는 불일치"


def test_readme_error_analysis_count_sync(readme_content, track_a_data):
    """README에 기재된 오차 분석 케이스 수가 실제 추출된 건수(37건)와 일치하는지 검증."""
    misclassified_count = track_a_data["error_analysis"]["total_misclassified"]
    assert misclassified_count >= 30, f"과제 요구사항 위반: 최소 30건 이상이어야 하나 {misclassified_count}건임"
    assert f"{misclassified_count}건" in readme_content or f"{misclassified_count} 건" in readme_content


def test_required_artifacts_and_figures_exist():
    """필수 4종 리포트와 4종 시각화 그래프 파일이 실제 존재하고 정상 크기(>0 바이트)인지 검증."""
    required_reports = [
        os.path.join(REPO_ROOT, "reports", "diagnosis_report.md"),
        os.path.join(REPO_ROOT, "track_a", "reports", "error_analysis_table.md"),
        os.path.join(REPO_ROOT, "track_b", "reports", "baseline_comparison_report.md"),
        os.path.join(REPO_ROOT, "reports", "data_leakage_evidence.md"),
    ]
    for path in required_reports:
        assert os.path.exists(path), f"필수 리포트 파일 누락: {path}"
        assert os.path.getsize(path) > 200, f"리포트 내용이 비어있거나 너무 작음: {path}"

    required_figures = [
        os.path.join(REPO_ROOT, "reports", "figures", "train_val_loss_curves.png"),
        os.path.join(REPO_ROOT, "track_b", "reports", "figures", "timeseries_prediction_vs_actual.png"),
        os.path.join(REPO_ROOT, "track_b", "reports", "figures", "baseline_comparison_barplot.png"),
        os.path.join(REPO_ROOT, "track_a", "reports", "figures", "error_analysis_gallery.png"),
    ]
    for path in required_figures:
        assert os.path.exists(path), f"필수 시각화 그래프 파일 누락: {path}"
        assert os.path.getsize(path) > 10000, f"이미지 파일 크기가 비정상(손상 의심): {path}"
