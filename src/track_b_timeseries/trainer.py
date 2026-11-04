"""Track B 시계열 모델 학습 및 평가 엔진 (PyTorch).

Train/Validation Loss 이력을 기록하여 편향/분산(Bias/Variance) 상태를 진단할 수 있도록 지원합니다.
"""

from typing import Dict, Any, List
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class TimeSeriesTrainer:
    """시계열 신경망 모델 학습자."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "validation_loss": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            t_loss = self.train_epoch(train_loader)
            v_loss = self.evaluate(val_loader)

            history["train_loss"].append(t_loss)
            history["validation_loss"].append(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(tensor_X)
            return preds.cpu().numpy()
