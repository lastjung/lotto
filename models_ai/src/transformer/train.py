"""
로또 Transformer 모델 학습 스크립트

사용법:
    python train.py --lottery korea_645
    python train.py --lottery canada_649 --history_length 20 --epochs 100
"""

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from models_ai.src.transformer.lotto_transformer import create_model


def load_lottery_config(lottery_id: str) -> dict:
    """로또 설정 로드 (config/lotteries.json)"""
    config_path = PROJECT_ROOT / "config" / "lotteries.json"
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if lottery_id not in configs:
        raise ValueError(f"Unknown lottery: {lottery_id}. Available: {list(configs.keys())}")
    return configs[lottery_id]


def load_training_config() -> dict:
    """학습 설정 로드 (config/training_config.json)"""
    config_path = PROJECT_ROOT / "config" / "training_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class LottoDataset(Dataset):
    """로또 데이터셋"""
    
    def __init__(self, data_path: str, history_length: int = 10):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.draws = [d["numbers"] for d in data["draws"]]
        self.history_length = history_length
        
        # 시퀀스 생성: 이전 history_length 회차 -> 다음 회차
        self.sequences = []
        for i in range(len(self.draws) - history_length):
            input_seq = self.draws[i:i + history_length]
            target = self.draws[i + history_length]
            self.sequences.append((input_seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        
        # 번호 -> 인덱스 (1~N -> 0~N-1)
        input_tensor = torch.tensor(input_seq, dtype=torch.long) - 1
        target_tensor = torch.tensor(target, dtype=torch.long) - 1
        
        return input_tensor, target_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, ball_count):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)  # (batch, ball_count, ball_ranges)
        
        # Loss 계산 (각 위치별 CrossEntropy)
        loss = 0
        for i in range(ball_count):
            loss += criterion(outputs[:, i, :], targets[:, i])
        loss /= ball_count
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, ball_count):
    """평가: 정확도 계산"""
    model.eval()
    correct_per_position = [0] * ball_count
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)  # (batch, ball_count)
            
            for i in range(ball_count):
                correct_per_position[i] += (predictions[:, i] == targets[:, i]).sum().item()
            total += targets.size(0)
    
    accuracies = [c / total for c in correct_per_position]
    return accuracies


def train(lottery_id: str, **overrides):
    """모델 학습 메인 함수"""
    
    # 로또 설정 로드
    lottery_config = load_lottery_config(lottery_id)
    ball_range = lottery_config["ball_range"]
    ball_ranges = ball_range[1]  # max value
    ball_count = lottery_config["ball_count"]
    data_path = PROJECT_ROOT / lottery_config["data_file"]
    
    print(f"\n{'='*50}")
    print(f"🎱 {lottery_config['name']} 모델 학습")
    print(f"{'='*50}")
    print(f"ball_ranges: {ball_ranges}, ball_count: {ball_count}")
    
    # 학습 설정 로드
    training_config = load_training_config()
    
    # CLI 오버라이드 적용
    for key, value in overrides.items():
        if value is not None:
            training_config[key] = value
    
    history_length = training_config["history_length"]
    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    lr = training_config["learning_rate"]
    
    print(f"history_length: {history_length}, epochs: {epochs}")
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    dataset = LottoDataset(str(data_path), history_length)
    
    # Train/Val 분리 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 모델 설정
    model_config = {
        "ball_ranges": ball_ranges,
        "history_length": history_length,
        "ball_count": ball_count,
        "d_model": training_config.get("d_model", 64),
        "nhead": training_config.get("nhead", 4),
        "num_layers": training_config.get("num_layers", 2),
        "dim_feedforward": training_config.get("dim_feedforward", 128),
        "dropout": training_config.get("dropout", 0.1),
    }
    
    # 모델 생성
    model = create_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {total_params:,}")
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    # 저장 경로
    save_dir = PROJECT_ROOT / "models_ai" / "trained" / "transformer"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / f"{lottery_id}.pt"
    
    # 학습 루프
    print("\n학습 시작...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, ball_count)
        val_accs = evaluate(model, val_loader, device, ball_count)
        avg_acc = sum(val_accs) / ball_count
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {avg_acc:.2%}")
        
        # 최고 모델 저장
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "lottery_id": lottery_id,
                "epoch": epoch,
                "loss": train_loss
            }, model_save_path)
            print(f"  ✓ 모델 저장: {model_save_path}")
    
    print(f"\n✅ 학습 완료! 저장: {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description="로또 Transformer 모델 학습")
    parser.add_argument("--lottery", "-l", type=str, required=True,
                        help="로또 ID (예: korea_645, canada_649)")
    parser.add_argument("--history_length", type=int, default=None,
                        help="입력 회차 수 (기본: training_config.json)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="학습률")
    
    args = parser.parse_args()
    
    train(
        lottery_id=args.lottery,
        history_length=args.history_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
