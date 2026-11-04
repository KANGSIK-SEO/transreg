"""Track B 시계열 딥러닝 모델 모듈 (LSTM, GRU, Attention-LSTM)."""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """LSTM 기반 시계열 주가 예측 모델."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.dropout(last_step)
        return self.fc(out).squeeze(-1)


class GRURegressor(nn.Module):
    """GRU 기반 시계열 주가 예측 모델."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        out = self.dropout(last_step)
        return self.fc(out).squeeze(-1)


class AttentionBlock(nn.Module):
    """시퀀스 각 시점의 은닉 상태에 대한 Self-Attention 가중치 계산."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        # rnn_outputs shape: (batch, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch, hidden_dim)
        return context


class AttentionLSTMRegressor(nn.Module):
    """Attention 메커니즘이 결합된 LSTM 모델 (보너스 과제)."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionBlock(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        context = self.attention(lstm_out)  # (batch, hidden_dim)
        out = self.dropout(context)
        return self.fc(out).squeeze(-1)
