#!/usr/bin/env python3
"""
로또 모델 학습 - 총괄 스크립트
다양한 모델 타입 및 로또 종류를 선택해서 학습 실행
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="로또 AI 모델 학습기")
    parser.add_argument(
        "--model", "-m",
        choices=["transformer", "lstm"],
        default="transformer",
        help="학습할 모델 타입 선택 (기본: transformer)"
    )
    parser.add_argument(
        "--lottery", "-l",
        choices=SUPPORTED_LOTTERIES + ["all"],
        default="korea_645",
        help="학습할 로또 종류 (기본: korea_645, all=전체 학습)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="학습 에폭 수 (기본: 50)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="배치 사이즈 (기본: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률 (기본: 0.001)"
    )
    
    args = parser.parse_args()
    
    # 학습할 로또 목록 결정
    lotteries = SUPPORTED_LOTTERIES if args.lottery == "all" else [args.lottery]
    
    for lottery_id in lotteries:
        print(f"\n{'='*50}")
        print(f"🚀 {lottery_id} - {args.model.upper()} 모델 학습 시작...")
        print(f"{'='*50}")
        
        # 데이터 경로
        data_path = PROJECT_ROOT / "data" / lottery_id / "draws.json"
        if not data_path.exists():
            print(f"⚠️ 데이터 파일이 없습니다: {data_path}")
            continue
        
        # 모델 저장 경로 (trained 폴더)
        model_save_path = PROJECT_ROOT / "lotto_models" / "trained" / args.model / f"{lottery_id}.pt"
        
        if args.model == "transformer":
            from lotto_models.src.transformer.train import train as transformer_train
            transformer_train(
                data_path=str(data_path),
                model_save_path=str(model_save_path),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
        elif args.model == "lstm":
            from lotto_models.src.lstm.train import train as lstm_train
            lstm_train(
                data_path=str(data_path),
                model_save_path=str(model_save_path),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
        
        print(f"✅ {lottery_id} 모델 저장: {model_save_path}")

    print(f"\n{'='*50}")
    print("🎉 전체 학습 완료!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
