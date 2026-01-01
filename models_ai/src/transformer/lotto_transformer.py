"""
로또 번호 예측을 위한 소형 Transformer 모델
학습/엔터테인먼트 목적 - 당첨 보장 없음
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """시퀀스 위치 정보를 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LottoTransformer(nn.Module):
    """
    로또 번호 예측용 소형 Transformer
    
    입력: 이전 N회차의 번호들 (N x ball_count)
    출력: 다음 회차 각 번호(1~ball_ranges)의 확률
    """
    
    def __init__(
        self,
        ball_ranges: int = 45,        # 로또 번호 범위 (1~45)
        history_length: int = 10,     # 입력 시퀀스 길이 (이전 N회차)
        ball_count: int = 6,          # 공 개수
        d_model: int = 64,            # 임베딩 차원
        nhead: int = 4,               # 어텐션 헤드 수
        num_layers: int = 2,          # Transformer 레이어 수
        dim_feedforward: int = 128,   # FFN 차원
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ball_ranges = ball_ranges
        self.history_length = history_length
        self.ball_count = ball_count
        self.d_model = d_model
        
        # 번호 임베딩 (1~ball_ranges -> d_model 차원)
        self.number_embedding = nn.Embedding(ball_ranges + 1, d_model)  # 0: padding
        
        # 위치 임베딩 (ball_count개 번호 위치)
        self.position_in_draw = nn.Embedding(ball_count, d_model)
        
        # 회차 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_len=history_length * ball_count)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 레이어: 각 번호의 확률
        self.fc_out = nn.Linear(d_model, ball_ranges)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, 6) - 이전 seq_length 회차의 번호들
        
        Returns:
            (batch, 6, num_numbers) - 다음 회차 6개 번호 각각의 확률
        """
        batch_size = x.size(0)
        
        # (batch, seq_length, 6) -> (batch, seq_length * 6)
        x_flat = x.view(batch_size, -1)
        
        # 번호 임베딩
        embedded = self.number_embedding(x_flat)  # (batch, seq*6, d_model)
        
        # 번호 위치 임베딩 (0~ball_count-1 반복)
        positions = torch.arange(self.ball_count).repeat(self.history_length).to(x.device)
        pos_embedded = self.position_in_draw(positions)
        embedded = embedded + pos_embedded
        
        # 회차 위치 인코딩
        embedded = self.positional_encoding(embedded)
        embedded = self.dropout(embedded)
        
        # Transformer
        encoded = self.transformer(embedded)  # (batch, seq*6, d_model)
        
        # 마지막 ball_count개 토큰의 출력을 사용 (다음 회차 예측)
        last_outputs = encoded[:, -self.ball_count:, :]  # (batch, ball_count, d_model)
        
        # 각 위치에서 번호 확률 예측
        logits = self.fc_out(last_outputs)  # (batch, 6, num_numbers)
        
        return logits
    
    def predict(self, x, temperature: float = 1.0, top_k: int = 10):
        """
        번호 생성 (샘플링)
        
        Args:
            x: 입력 시퀀스
            temperature: 샘플링 온도 (높을수록 다양함)
            top_k: 상위 k개 번호에서만 샘플링
        
        Returns:
            (batch, 6) - 생성된 번호
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (batch, 6, 45)
            
            # Temperature 적용
            logits = logits / temperature
            
            # Top-k 필터링
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # 확률로 변환
            probs = F.softmax(logits, dim=-1)
            
            # 샘플링
            predicted = []
            for i in range(self.ball_count):
                sampled = torch.multinomial(probs[:, i, :], 1)
                predicted.append(sampled + 1)  # 0-indexed -> 1-indexed
            
            return torch.cat(predicted, dim=1)


def create_model(config: dict = None) -> LottoTransformer:
    """모델 생성 헬퍼 함수"""
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
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1
    }
    
    if config:
        default_config.update(config)
    
    return LottoTransformer(**default_config)


if __name__ == "__main__":
    # 테스트
    model = create_model()
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    print(f"모델 크기 추정: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 더미 입력 테스트
    dummy_input = torch.randint(1, 46, (2, 10, 6))  # batch=2, seq=10, numbers=6
    output = model(dummy_input)
    print(f"입력 shape: {dummy_input.shape}")
    print(f"출력 shape: {output.shape}")
    
    # 예측 테스트
    predicted = model.predict(dummy_input)
    print(f"예측 번호: {predicted}")
