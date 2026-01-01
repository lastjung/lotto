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
    
    입력: 이전 N회차의 번호들 (N x ball_count)
    출력: 다음 회차 각 번호(1~ball_ranges)의 확률
    """
    
    def __init__(
        self,
        ball_ranges: int = 45,
        history_length: int = 10,
        ball_count: int = 6,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.ball_ranges = ball_ranges
        self.history_length = history_length
        self.ball_count = ball_count
        
        # 번호 임베딩
        self.embedding = nn.Embedding(ball_ranges + 1, embedding_dim)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim * ball_count,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_dim, ball_ranges * ball_count)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, history_length, ball_count)
        Returns:
            (batch, ball_count, ball_ranges)
        """
        batch_size = x.size(0)
        
        # 임베딩 및 결합: (batch, seq, ball_count, emb) -> (batch, seq, ball_count * emb)
        embedded = self.embedding(x)  # (batch, seq, ball_count, embedding_dim)
        embedded = embedded.view(batch_size, self.history_length, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden_dim)
        
        # 마지막 시점의 출력 사용
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        last_out = self.dropout(last_out)
        
        # 로짓 생성: (batch, hidden_dim) -> (batch, ball_count * ball_ranges)
        logits = self.fc_out(last_out)
        logits = logits.view(batch_size, self.ball_count, self.ball_ranges)
        
        return logits

    def predict(self, x, temperature: float = 1.0, top_k: int = 10):
        """번호 생성"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (batch, ball_count, ball_ranges)
            logits = logits / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            
            predicted = []
            for i in range(self.ball_count):
                sampled = torch.multinomial(probs[:, i, :], 1)
                predicted.append(sampled + 1)
            
            return torch.cat(predicted, dim=1)


def create_model(config: dict = None) -> LottoLSTM:
    # 이전 변수명 호환성 (num_numbers→ball_ranges, seq_length→history_length)
    if config:
        if "num_numbers" in config and "ball_ranges" not in config:
            config["ball_ranges"] = config.pop("num_numbers")
        if "seq_length" in config and "history_length" not in config:
            config["history_length"] = config.pop("seq_length")
    
    default_config = {
        "ball_ranges": 45,
        "history_length": 10,
        "ball_count": 6,
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2
    }
    if config:
        default_config.update(config)
    return LottoLSTM(**default_config)

