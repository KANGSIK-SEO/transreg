"""Track A 전이학습 모델 트레이너 모듈.

Train/Validation 성능 이력을 추적하여 편향/분산 진단 근거를 생성합니다.
"""

from typing import Dict, List, Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TransferTrainer:
    """전이학습 트레이너 클래스."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # requires_grad=True인 파라미터만 옵티마이저에 전달
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "train_acc": [],
            "validation_loss": [],
            "validation_acc": [],
        }

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            t_loss, t_acc = self.train_epoch(train_loader)
            v_loss, v_acc = self.evaluate(val_loader)

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["validation_loss"].append(v_loss)
            history["validation_acc"].append(v_acc)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
