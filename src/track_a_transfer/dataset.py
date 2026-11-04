"""Track A Few-shot 이미지 데이터셋 모듈.

요구사항:
- 데이터가 적은(클래스당 50장 미만) 환경 구성
- CIFAR-10 중 5개 클래스 선별: ['airplane', 'automobile', 'bird', 'cat', 'dog']
- 각 클래스당 Train 35장 (< 50장), Validation 10장, Test 15장 구성
- 문서/코드 용어: Train, Validation, Test 통일
"""

from typing import Tuple, List, Dict
import os
import ssl
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# macOS urllib SSL 인증서 검증 우회 설정
ssl._create_default_https_context = ssl._create_unverified_context



SELECTED_CLASSES = [0, 1, 2, 3, 5]  # airplane, automobile, bird, cat, dog
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "dog"]
CLASS_MAP = {orig: new for new, orig in enumerate(SELECTED_CLASSES)}


class FewShotImageDataset:
    """Few-shot 이미지 데이터셋 빌더 (클래스당 50장 미만 엄격 준수)."""

    def __init__(
        self,
        data_root: str = "data/downloaded",
        train_per_class: int = 35,
        val_per_class: int = 10,
        test_per_class: int = 25,
        img_size: int = 224,
    ):
        assert train_per_class < 50, "과제 요구사항: 클래스당 50장 미만이어야 합니다."
        self.data_root = data_root
        self.train_per_class = train_per_class
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.img_size = img_size

        # 사전학습 ResNet 입력 규격 (ImageNet 통계)
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 원본 시각화용 transform (Denormalization)
        self.raw_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def load_cifar_subsets(
        self,
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
        """CIFAR-10 다운로드 및 5개 클래스에 대해 Few-shot 분할 수행.

        Returns:
            train_dataset, validation_dataset, test_dataset, raw_test_dataset (시각화용)
        """
        os.makedirs(self.data_root, exist_ok=True)

        full_train = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=self.train_transform
        )
        full_val = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=self.eval_transform
        )
        full_test = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=self.eval_transform
        )
        full_raw_test = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=self.raw_transform
        )

        # 인덱스 필터링 및 서브셋 샘플링
        train_indices = []
        val_indices = []
        test_indices = []

        np.random.seed(42)

        # Train & Validation 인덱스 추출 (train split에서)
        for orig_cls in SELECTED_CLASSES:
            cls_indices = [
                i for i, label in enumerate(full_train.targets) if label == orig_cls
            ]
            shuffled = np.random.permutation(cls_indices)
            train_indices.extend(shuffled[: self.train_per_class])
            val_indices.extend(
                shuffled[self.train_per_class : self.train_per_class + self.val_per_class]
            )

        # Test 인덱스 추출 (test split에서)
        for orig_cls in SELECTED_CLASSES:
            cls_indices = [
                i for i, label in enumerate(full_test.targets) if label == orig_cls
            ]
            shuffled = np.random.permutation(cls_indices)
            test_indices.extend(shuffled[: self.test_per_class])

        # 라벨 리매핑 래퍼 데이터셋
        class RemappedSubset(Dataset):
            def __init__(self, subset: Dataset):
                self.subset = subset

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                img, orig_label = self.subset[idx]
                return img, CLASS_MAP[orig_label]

        train_ds = RemappedSubset(Subset(full_train, train_indices))
        val_ds = RemappedSubset(Subset(full_val, val_indices))
        test_ds = RemappedSubset(Subset(full_test, test_indices))
        raw_test_ds = RemappedSubset(Subset(full_raw_test, test_indices))

        return train_ds, val_ds, test_ds, raw_test_ds
