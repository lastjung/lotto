"""
로또 번호 생성 스크립트 (멀티 로또 지원)
학습된 모델을 사용하여 번호를 추천합니다.

Usage:
    python generate.py --lottery korea_645
    python generate.py --lottery usa_powerball
"""

import json
import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from models_ai.src.transformer.lotto_transformer import create_model, LottoTransformer

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_lottery_config(lottery_id: str) -> dict:
    """로또 설정 로드"""
    config_path = PROJECT_ROOT / "config" / "lotteries.json"
    default = {"ball_count": 6, "ball_range": [1, 45]}
    
    if not config_path.exists():
        return default
    
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    return configs.get(lottery_id, default)


def load_model(lottery_id: str) -> LottoTransformer:
    """저장된 모델 로드"""
    model_path = PROJECT_ROOT / f"models_ai/trained/transformer/{lottery_id}.pt"
    
    # Fallback to default model
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models_ai/trained/transformer/lotto_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    
    config = checkpoint.get("config", {})
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✓ 모델 로드 완료 (Epoch: {checkpoint.get('epoch', '?')}, Lottery: {lottery_id})")
    return model


def get_recent_draws(lottery_id: str = "korea_645", n: int = 10) -> list:
    """최근 N회차 데이터 가져오기 (멀티 로또 지원)"""
    data_path = PROJECT_ROOT / f"data/{lottery_id}/draws.json"
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    draws = [d["numbers"] for d in data["draws"][-n:]]
    return draws


def generate_numbers(
    model: LottoTransformer,
    recent_draws: list,
    count: int = 5,
    temperature: float = 1.0,
    top_k: int = 15,
    ball_count: int = 6
) -> list:
    """번호 생성 (멀티 로또 지원)"""
    model.eval()
    
    # 입력 준비
    input_tensor = torch.tensor([recent_draws], dtype=torch.long)
    
    generated = []
    for _ in range(count):
        prediction = model.predict(input_tensor, temperature=temperature, top_k=top_k)
        numbers = sorted(prediction[0].tolist())
        
        # 중복 제거 (재생성) - 동적 ball_count
        if len(set(numbers)) == ball_count and numbers not in generated:
            generated.append(numbers)
    
    return generated


def analyze_numbers(numbers: list, max_num: int = 45) -> dict:
    """번호 조합 분석 (멀티 로또 지원)"""
    mid_point = max_num // 2
    n = len(numbers)
    return {
        "numbers": numbers,
        "sum": sum(numbers),
        "odd_count": sum(1 for num in numbers if num % 2 == 1),
        "even_count": sum(1 for num in numbers if num % 2 == 0),
        "low_count": sum(1 for num in numbers if num <= mid_point),
        "high_count": sum(1 for num in numbers if num > mid_point),
        "last_digits": [num % 10 for num in numbers],
        "deltas": [numbers[i+1] - numbers[i] for i in range(n - 1)]
    }


def main():
    parser = argparse.ArgumentParser(description="AI 로또 번호 생성기")
    parser.add_argument("--lottery", type=str, default="korea_645",
                        help="로또 ID (예: korea_645, usa_powerball, canada_649)")
    parser.add_argument("--count", type=int, default=5, help="생성할 게임 수")
    args = parser.parse_args()
    
    lottery_id = args.lottery
    lottery_config = load_lottery_config(lottery_id)
    ball_count = lottery_config.get("ball_count", 6)
    max_num = lottery_config.get("ball_range", [1, 45])[1]
    
    print("=" * 50)
    print(f"🎱 AI 로또 번호 생성기 - {lottery_id}")
    print(f"   (공 {ball_count}개, 범위 1-{max_num})")
    print("⚠️  엔터테인먼트 목적 - 당첨 보장 없음")
    print("=" * 50)
    
    # 모델 로드
    try:
        model = load_model(lottery_id)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   먼저 실행: python scripts/train.py --model transformer --lottery " + lottery_id)
        return
    
    # 최근 데이터 로드
    try:
        recent = get_recent_draws(lottery_id, n=10)
    except FileNotFoundError:
        print(f"\n❌ 데이터 파일을 찾을 수 없습니다: data/{lottery_id}/draws.json")
        return
    
    print(f"\n최근 10회차 기준으로 분석...")
    
    # 번호 생성
    print("\n🔮 AI 추천 번호:")
    print("-" * 40)
    
    generated = generate_numbers(model, recent, count=args.count, ball_count=ball_count)
    
    for i, numbers in enumerate(generated, 1):
        analysis = analyze_numbers(numbers, max_num=max_num)
        print(f"\n  #{i}: {numbers}")
        print(f"      합계: {analysis['sum']} | "
              f"홀짝: {analysis['odd_count']}:{analysis['even_count']} | "
              f"고저: {analysis['high_count']}:{analysis['low_count']}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
