"""Track A ResNet-18 기반 전이학습 모델 모듈.

실험 구성:
1. Linear Probing: Backbone 가중치 고정(Freeze) 후 최종 FC 분류기만 학습
2. Fine-tuning: 전체 레이어 가중치 미세 조정 (End-to-End Fine-tuning)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


def build_transfer_model(
    num_classes: int = 5,
    mode: str = "linear_probing",
    pretrained: bool = True,
) -> nn.Module:
    """ResNet-18 기반 전이학습 모델 생성.

    Args:
        num_classes: 분류 클래스 수 (기본 5개)
        mode: 'linear_probing' 또는 'fine_tuning'
        pretrained: ImageNet 사전학습 가중치 사용 여부
    """
    assert mode in ["linear_probing", "fine_tuning"], f"알 수 없는 모드: {mode}"

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if mode == "linear_probing":
        # 백본 파라미터 Freeze
        for param in model.parameters():
            param.requires_grad = False
    elif mode == "fine_tuning":
        # 전체 파라미터 학습 허용
        for param in model.parameters():
            param.requires_grad = True

    # 최종 분류기 FC 레이어 교체 (새로운 레이어는 자동으로 requires_grad=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_trainable_parameters_count(model: nn.Module) -> Tuple[int, int]:
    """(학습 가능 파라미터 수, 전체 파라미터 수) 반환."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
