"""
로또 번호 예측을 위한 LSTM 모델
학습/엔터테인먼트 목적 - 당첨 보장 없음
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LottoLSTM(nn.Module):
    """
    로또 번호 예측용 LSTM 모델
    
    입력: 이전 N회차의 번호들 (N x 6)
    출력: 다음 회차 각 번호(1~45)의 확률
    """
    
    def __init__(
        self,
        num_numbers: int = 45,
        seq_length: int = 10,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_numbers = num_numbers
        self.seq_length = seq_length
        
        # 번호 임베딩
        self.embedding = nn.Embedding(num_numbers + 1, embedding_dim)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 6,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_dim, num_numbers * 6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, 6)
        Returns:
            (batch, 6, num_numbers)
        """
        batch_size = x.size(0)
        
        # 임베딩 및 결합: (batch, seq, 6, emb) -> (batch, seq, 6 * emb)
        embedded = self.embedding(x)  # (batch, seq, 6, embedding_dim)
        embedded = embedded.view(batch_size, self.seq_length, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden_dim)
        
        # 마지막 시점의 출력 사용
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        last_out = self.dropout(last_out)
        
        # 로짓 생성: (batch, hidden_dim) -> (batch, 6 * num_numbers)
        logits = self.fc_out(last_out)
        logits = logits.view(batch_size, 6, self.num_numbers)
        
        return logits

    def predict(self, x, temperature: float = 1.0, top_k: int = 10):
        """번호 생성"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (batch, 6, 45)
            logits = logits / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            
            predicted = []
            for i in range(6):
                sampled = torch.multinomial(probs[:, i, :], 1)
                predicted.append(sampled + 1)
            
            return torch.cat(predicted, dim=1)


def create_model(config: dict = None) -> LottoLSTM:
    default_config = {
        "num_numbers": 45,
        "seq_length": 10,
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2
    }
    if config:
        default_config.update(config)
    return LottoLSTM(**default_config)
