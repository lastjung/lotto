#!/usr/bin/env python3
"""
로또 번호 생성 - 총괄 스크립트 (멀티 로또 지원)
다양한 모델 타입과 로또 종류를 선택해서 번호 생성

Usage:
    python generate.py --model transformer --lottery korea_645
    python generate.py --model lstm --lottery usa_powerball --count 3
"""

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 지원하는 로또 목록
SUPPORTED_LOTTERIES = [
    "korea_645",
    "usa_powerball",
    "usa_megamillions",
    "canada_649",
    "japan_loto6"
]


def load_lottery_config(lottery_id: str) -> dict:
    """로또 설정 로드"""
    config_path = PROJECT_ROOT / "config" / "lotteries.json"
    default = {"ball_count": 6, "ball_range": [1, 45]}
    
    if not config_path.exists():
        return default
    
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    return configs.get(lottery_id, default)


def run_transformer(lottery_id: str, count: int):
    """Transformer 모델로 번호 생성"""
    from models_ai.src.transformer.generate import (
        load_model, get_recent_draws, generate_numbers, analyze_numbers, load_lottery_config
    )
    
    lottery_config = load_lottery_config(lottery_id)
    ball_count = lottery_config.get("ball_count", 6)
    max_num = lottery_config.get("ball_range", [1, 45])[1]
    
    print("=" * 50)
    print(f"🎱 AI 로또 번호 생성기 (Transformer) - {lottery_id}")
    print(f"   (공 {ball_count}개, 범위 1-{max_num})")
    print("⚠️  엔터테인먼트 목적 - 당첨 보장 없음")
    print("=" * 50)
    
    try:
        model = load_model(lottery_id)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"   먼저 실행: python scripts/train.py --model transformer --lottery {lottery_id}")
        return
    
    try:
        recent = get_recent_draws(lottery_id, n=10)
    except FileNotFoundError:
        print(f"\n❌ 데이터 파일을 찾을 수 없습니다: data/{lottery_id}/draws.json")
        return
    
    print(f"\n최근 10회차 기준으로 분석...")
    print("\n🔮 AI 추천 번호:")
    print("-" * 40)
    
    generated = generate_numbers(model, recent, count=count, ball_count=ball_count)
    
    for i, numbers in enumerate(generated, 1):
        analysis = analyze_numbers(numbers, max_num=max_num)
        print(f"\n  #{i}: {numbers}")
        print(f"      합계: {analysis['sum']} | "
              f"홀짝: {analysis['odd_count']}:{analysis['even_count']} | "
              f"고저: {analysis['high_count']}:{analysis['low_count']}")
    
    print("\n" + "=" * 50)


def run_lstm(lottery_id: str, count: int):
    """LSTM 모델로 번호 생성"""
    from models_ai.src.lstm.generate import (
        load_model, get_recent_draws, generate_numbers, analyze_numbers, load_lottery_config
    )
    
    lottery_config = load_lottery_config(lottery_id)
    ball_count = lottery_config.get("ball_count", 6)
    max_num = lottery_config.get("ball_range", [1, 45])[1]
    
    print("=" * 50)
    print(f"🎱 AI 로또 번호 생성기 (LSTM) - {lottery_id}")
    print(f"   (공 {ball_count}개, 범위 1-{max_num})")
    print("⚠️  엔터테인먼트 목적 - 당첨 보장 없음")
    print("=" * 50)
    
    try:
        model = load_model(lottery_id)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"   먼저 실행: python scripts/train.py --model lstm --lottery {lottery_id}")
        return
    
    try:
        recent = get_recent_draws(lottery_id, n=10)
    except FileNotFoundError:
        print(f"\n❌ 데이터 파일을 찾을 수 없습니다: data/{lottery_id}/draws.json")
        return
    
    print(f"\n최근 10회차 기준으로 분석...")
    print("\n🔮 AI 추천 번호:")
    print("-" * 40)
    
    generated = generate_numbers(model, recent, count=count, ball_count=ball_count)
    
    for i, numbers in enumerate(generated, 1):
        analysis = analyze_numbers(numbers, max_num=max_num)
        print(f"\n  #{i}: {numbers}")
        print(f"      합계: {analysis['sum']} | "
              f"홀짝: {analysis['odd_count']}:{analysis['even_count']} | "
              f"고저: {analysis['high_count']}:{analysis['low_count']}")
    
    print("\n" + "=" * 50)


def run_vector(lottery_id: str, count: int):
    """Vector 모델로 번호 생성"""
    from models_ai.src.vector.lotto_vector import LottoVectorModel
    
    lottery_config = load_lottery_config(lottery_id)
    ball_count = lottery_config.get("ball_count", 6)
    ball_range = tuple(lottery_config.get("ball_range", [1, 45]))
    max_num = ball_range[1]
    
    print("=" * 50)
    print(f"🎱 AI 로또 번호 생성기 (Vector) - {lottery_id}")
    print(f"   (공 {ball_count}개, 범위 1-{max_num})")
    print("⚠️  엔터테인먼트 목적 - 당첨 보장 없음")
    print("=" * 50)
    
    # 데이터 로드
    data_path = PROJECT_ROOT / f"data/{lottery_id}/draws.json"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        draws = [d["numbers"] for d in data["draws"]]
    except FileNotFoundError:
        print(f"\n❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 모델 학습 및 생성
    print("\n모델 학습 중...")
    model = LottoVectorModel(
        ball_count=ball_count,
        ball_range=ball_range,
        n_clusters=5,
        n_neighbors=10
    )
    model.fit(draws)
    
    print("\n🔮 AI 추천 번호:")
    print("-" * 40)
    
    results = model.generate(count=count)
    
    for i, r in enumerate(results, 1):
        numbers = r['numbers']
        print(f"\n  #{i}: {numbers}")
        print(f"      유사도: {r['similarity']:.1%} | AC값: {r['ac_value']} | 클러스터: {r['cluster']}")
    
    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(description="로또 AI 번호 생성기 (멀티 로또 지원)")
    parser.add_argument(
        "--model", "-m",
        choices=["transformer", "lstm", "vector"],
        default="transformer",
        help="모델 타입 선택 (기본: transformer)"
    )
    parser.add_argument(
        "--lottery", "-l",
        choices=SUPPORTED_LOTTERIES,
        default="korea_645",
        help="로또 종류 (기본: korea_645)"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="생성할 번호 세트 수 (기본: 5)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 모델/로또 목록"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 사용 가능한 모델:")
        print("  - transformer: Transformer 기반 모델 (딥러닝)")
        print("  - lstm: LSTM 기반 모델 (딥러닝)")
        print("  - vector: Vector 클러스터링 기반 모델 (ML)")
        print("\n🌍 지원하는 로또:")
        for lid in SUPPORTED_LOTTERIES:
            config = load_lottery_config(lid)
            print(f"  - {lid}: {config.get('name', lid)}")
        return
    
    if args.model == "transformer":
        run_transformer(args.lottery, args.count)
    elif args.model == "lstm":
        run_lstm(args.lottery, args.count)
    elif args.model == "vector":
        run_vector(args.lottery, args.count)


if __name__ == "__main__":
    main()
