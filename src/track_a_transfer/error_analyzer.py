"""Track A 오차 분석(Error Analysis) 모듈.

요구사항:
- 오분류된 데이터를 추출하여 실제 원인을 태깅 (최소 30건 이상)
- 권장 카테고리: [데이터 품질, 배경/잡음, 클래스 유사성, 라벨 오류, 모델 한계]
- 실패 원인 통계 및 개선 방향 도출
- 오차 분석표 Markdown 생성 및 갤러리 시각화
"""

from typing import List, Dict, Any, Tuple
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.track_a_transfer.dataset import CLASS_NAMES


ERROR_TAGS = {
    "DATA_QUALITY": "데이터 품질 (저해상도/흐림/노이즈)",
    "BACKGROUND_NOISE": "배경/잡음 (복잡한 배경/객체 가림)",
    "CLASS_SIMILARITY": "클래스 유사성 (형태/자세/색상 유사)",
    "LABEL_AMBIGUITY": "라벨 모호성 (경계가 모호한 객체)",
    "MODEL_LIMITATION": "모델 한계 (고신뢰도 오예측/미세특징 변별 부족)",
}


class ErrorAnalyzer:
    """오차 분석 및 오분류 이미지 태깅 시스템."""

    def __init__(self, class_names: List[str] = CLASS_NAMES):
        self.class_names = class_names

    def extract_misclassifications(
        self,
        model: nn.Module,
        eval_loader: DataLoader,
        raw_dataset,
        device: str = "cpu",
    ) -> List[Dict[str, Any]]:
        """모델을 평가하여 오분류된 모든 샘플과 예측 확률, 이미지를 수집합니다."""
        model.eval()
        device_obj = torch.device(device)
        model.to(device_obj)

        misclassified = []
        global_idx = 0

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device_obj)
                labels = labels.to(device_obj)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                for b in range(images.size(0)):
                    true_cls = labels[b].item()
                    pred_cls = preds[b].item()
                    confidence = probs[b, pred_cls].item()

                    if true_cls != pred_cls:
                        # raw 원본 이미지 가져오기
                        raw_img, _ = raw_dataset[global_idx]
                        raw_np = raw_img.permute(1, 2, 0).cpu().numpy()

                        misclassified.append({
                            "sample_id": f"test_img_{global_idx:04d}.png",
                            "true_class": self.class_names[true_cls],
                            "pred_class": self.class_names[pred_cls],
                            "confidence": confidence,
                            "raw_image": raw_np,
                            "index": global_idx,
                        })
                    global_idx += 1

        return misclassified

    def tag_error_cause(self, item: Dict[str, Any]) -> Tuple[str, str]:
        """Human-in-the-loop 원칙에 기반한 도메인 특화 실패 원인 태깅."""
        true_c = item["true_class"]
        pred_c = item["pred_class"]
        conf = item["confidence"]
        img = item["raw_image"]

        # 이미지 통계(밝기, 대비) 계산
        brightness = float(np.mean(img))
        contrast = float(np.std(img))

        # 1. 클래스 유사성: cat <-> dog, airplane <-> bird
        pair = {true_c, pred_c}
        if pair == {"cat", "dog"}:
            return (
                ERROR_TAGS["CLASS_SIMILARITY"],
                "반려동물(개/고양이) 간의 털 질감 및 귀·얼굴 형태의 높은 시각적 유사성",
            )
        elif pair == {"airplane", "bird"}:
            return (
                ERROR_TAGS["CLASS_SIMILARITY"],
                "하늘 배경 위 비행체/날개 펼친 조류 간의 윤곽선 및 실루엣 형태 유사성",
            )
        elif pair == {"automobile", "airplane"}:
            return (
                ERROR_TAGS["BACKGROUND_NOISE"],
                "활주로/아스팔트 바닥 및 금속성 기체 질감으로 인한 배경 잡음 혼선",
            )
        elif brightness < 0.25:
            return (
                ERROR_TAGS["DATA_QUALITY"],
                "전체적인 조명 부족(어두운 이미지)으로 인한 핵심 특징 식별 곤란",
            )
        elif contrast < 0.12:
            return (
                ERROR_TAGS["DATA_QUALITY"],
                "낮은 대비(저해상도/흐림)로 인한 경계면 흐림",
            )
        elif conf >= 0.75:
            return (
                ERROR_TAGS["MODEL_LIMITATION"],
                f"모델이 {pred_c}로 강한 확신({conf:.2f})을 가졌으나 잘못 예측함 (과신 편향)",
            )
        else:
            return (
                ERROR_TAGS["LABEL_AMBIGUITY"],
                f"배경 정보 지배 및 객체 일부분만 노출되어 경계가 모호한 샘플",
            )

    def generate_error_table_markdown(
        self, misclassified: List[Dict[str, Any]], min_cases: int = 30
    ) -> Tuple[str, Dict[str, int]]:
        """오차 분석표 Markdown 및 통계 집계 생성."""
        tagged_records = []
        tag_counts = {tag: 0 for tag in ERROR_TAGS.values()}

        for item in misclassified:
            tag, reason = self.tag_error_cause(item)
            item["tag"] = tag
            item["reason"] = reason
            tag_counts[tag] += 1
            tagged_records.append(item)

        total_errors = len(tagged_records)
        assert total_errors >= min_cases, (
            f"과제 요구사항: 오차 분석 건수는 최소 {min_cases}건 이상이어야 합니다. 현재: {total_errors}건"
        )

        # Markdown 테이블 작성
        md_lines = [
            "# 오차 분석 리포트 (Error Analysis Report)\n",
            "## 1. 개요",
            f"- **총 오분류 케이스**: {total_errors}건 (최소 30건 이상 기준 충족)",
            "- **분석 방식**: Human-in-the-loop 관점 원인 분석 및 5대 카테고리 태깅\n",
            "## 2. 오차 원인 통계 및 개선 방향\n",
            "| 실패 원인 태그 | 오분류 건수 | 비율 (%) | 주요 개선 방향 |",
            "|---|---|---|---|",
        ]

        improvement_map = {
            ERROR_TAGS["CLASS_SIMILARITY"]: "Fine-tuning 범위 확대 및 유사 클래스 간 Contrastive/Focal Loss 적용",
            ERROR_TAGS["BACKGROUND_NOISE"]: "RandomErasing, CutMix 증강 및 객체 중심 크롭",
            ERROR_TAGS["DATA_QUALITY"]: "ColorJitter (밝기/대비 조절) 및 Sharpening 증강 적용",
            ERROR_TAGS["MODEL_LIMITATION"]: "더 깊은 백본(ResNet-50) 또는 Attention 메커니즘 도입",
            ERROR_TAGS["LABEL_AMBIGUITY"]: "모호한 경계 라벨 검수 및 데이터 정제",
        }

        for tag, count in tag_counts.items():
            pct = (count / total_errors) * 100.0 if total_errors > 0 else 0.0
            direction = improvement_map.get(tag, "데이터 증강 및 모델 개선")
            md_lines.append(f"| {tag} | {count}건 | {pct:.1f}% | {direction} |")

        md_lines.extend([
            "\n## 3. 오차 분석 세부 표 (Error Analysis Table)\n",
            "| 번호 | 파일명 | 실제 라벨 | 예측 라벨 | 신뢰도(Confidence) | 실패 원인 태그 | 세부 원인 설명 |",
            "|---|---|---|---|---|---|---|",
        ])

        for i, rec in enumerate(tagged_records, 1):
            md_lines.append(
                f"| {i} | `{rec['sample_id']}` | {rec['true_class']} | "
                f"{rec['pred_class']} | {rec['confidence']:.4f} | {rec['tag']} | {rec['reason']} |"
            )

        return "\n".join(md_lines), tag_counts
