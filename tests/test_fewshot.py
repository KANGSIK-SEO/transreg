"""Tests for Track A Few-shot constraints."""

import pytest
from src.track_a_transfer.dataset import FewShotImageDataset, CLASS_NAMES


def test_fewshot_train_sample_count():
    ds_builder = FewShotImageDataset(train_per_class=35, val_per_class=10, test_per_class=20)
    assert ds_builder.train_per_class < 50, "과제 요구사항: 클래스당 50장 미만이어야 합니다."


def test_selected_classes_count():
    assert len(CLASS_NAMES) == 5
    assert "airplane" in CLASS_NAMES
    assert "cat" in CLASS_NAMES
    assert "dog" in CLASS_NAMES
