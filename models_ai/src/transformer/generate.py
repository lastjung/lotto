"""
로또 번호 생성 스크립트
학습된 모델을 사용하여 번호를 추천합니다.
"""

import json
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from models_ai.src.transformer.lotto_transformer import create_model, LottoTransformer


def load_model(model_path: str = "models_ai/trained/transformer/lotto_model.pt") -> LottoTransformer:
    """저장된 모델 로드"""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    
    config = checkpoint.get("config", {})
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✓ 모델 로드 완료 (Epoch: {checkpoint.get('epoch', '?')})")
    return model


def get_recent_draws(data_path: str = "data/korea_645/draws.json", n: int = 10) -> list:
    """최근 N회차 데이터 가져오기"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    draws = [d["numbers"] for d in data["draws"][-n:]]
    return draws


def generate_numbers(
    model: LottoTransformer,
    recent_draws: list,
    count: int = 5,
    temperature: float = 1.0,
    top_k: int = 15
) -> list:
    """번호 생성"""
    model.eval()
    
    # 입력 준비
    input_tensor = torch.tensor([recent_draws], dtype=torch.long)
    
    generated = []
    for _ in range(count):
        prediction = model.predict(input_tensor, temperature=temperature, top_k=top_k)
        numbers = sorted(prediction[0].tolist())
        
        # 중복 제거 (재생성)
        if len(set(numbers)) == 6 and numbers not in generated:
            generated.append(numbers)
    
    return generated


def analyze_numbers(numbers: list) -> dict:
    """번호 조합 분석"""
    return {
        "numbers": numbers,
        "sum": sum(numbers),
        "odd_count": sum(1 for n in numbers if n % 2 == 1),
        "even_count": sum(1 for n in numbers if n % 2 == 0),
        "low_count": sum(1 for n in numbers if n <= 22),  # 1-22
        "high_count": sum(1 for n in numbers if n > 22),   # 23-45
        "last_digits": [n % 10 for n in numbers],
        "deltas": [numbers[i+1] - numbers[i] for i in range(5)]
    }


def main():
    print("=" * 50)
    print("🎱 AI 로또 번호 생성기")
    print("⚠️  엔터테인먼트 목적 - 당첨 보장 없음")
    print("=" * 50)
    
    # 모델 로드
    model_path = Path("models_ai/trained/transformer/lotto_model.pt")
    if not model_path.exists():
        print("\n❌ 학습된 모델이 없습니다.")
        print("   먼저 실행: python scripts/train_model.py")
        return
    
    model = load_model(str(model_path))
    
    # 최근 데이터 로드
    recent = get_recent_draws(n=10)
    print(f"\n최근 10회차 기준으로 분석...")
    
    # 번호 생성
    print("\n🔮 AI 추천 번호:")
    print("-" * 40)
    
    generated = generate_numbers(model, recent, count=5)
    
    for i, numbers in enumerate(generated, 1):
        analysis = analyze_numbers(numbers)
        print(f"\n  #{i}: {numbers}")
        print(f"      합계: {analysis['sum']} | "
              f"홀짝: {analysis['odd_count']}:{analysis['even_count']} | "
              f"고저: {analysis['high_count']}:{analysis['low_count']}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
