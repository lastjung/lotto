"""
로또 LSTM 모델 학습 스크립트
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from models_ai.src.lstm.lotto_lstm import create_model


class LottoDataset(Dataset):
    """로또 데이터셋"""
    
    def __init__(self, data_path: str, seq_length: int = 10):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.draws = [d["numbers"] for d in data["draws"]]
        self.seq_length = seq_length
        
        # 시퀀스 생성: 이전 seq_length 회차 -> 다음 회차
        self.sequences = []
        for i in range(len(self.draws) - seq_length):
            input_seq = self.draws[i:i + seq_length]
            target = self.draws[i + seq_length]
            self.sequences.append((input_seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        
        # 번호 그대로 유지 (Model 내부에서 Embedding 처리)
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long) - 1  # 0-indexed for CrossEntropy
        
        return input_tensor, target_tensor


def train_epoch(model, dataloader, optimizer, criterion, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)  # (batch, 6, 45)
        
        # Loss 계산
        loss = 0
        for i in range(6):
            loss += criterion(outputs[:, i, :], targets[:, i])
        loss /= 6
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """평가"""
    model.eval()
    correct_per_position = [0] * 6
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            
            for i in range(6):
                correct_per_position[i] += (predictions[:, i] == targets[:, i]).sum().item()
            total += targets.size(0)
    
    accuracies = [c / total for c in correct_per_position]
    return accuracies


def train(
    data_path: str = "data/korea_645/draws.json",
    model_save_path: str = "models_ai/trained/lstm/lotto_model.pt",
    seq_length: int = 10,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = None
):
    """모델 학습 메인 함수"""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Device: {device}")
    
    # 데이터 로드
    print("데이터 로드 중...")
    dataset = LottoDataset(data_path, seq_length)
    
    # Train/Val 분리
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 모델 생성
    model_config = {
        "num_numbers": 45,
        "seq_length": seq_length,
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2
    }
    model = create_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {total_params:,}")
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    # 학습 루프
    print("\n학습 시작...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_accs = evaluate(model, val_loader, device)
        avg_acc = sum(val_accs) / 6
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Avg Val Acc: {avg_acc:.2%}")
        
        # 최고 모델 저장
        if train_loss < best_loss:
            best_loss = train_loss
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "epoch": epoch,
                "loss": train_loss
            }, model_save_path)
            print(f"  ✓ 모델 저장: {model_save_path}")
    
    print("\n✅ 학습 완료!")


if __name__ == "__main__":
    train()
