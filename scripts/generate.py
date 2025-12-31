#!/usr/bin/env python3
"""
로또 번호 생성 - 총괄 스크립트
다양한 모델 타입을 선택해서 번호 생성
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="로또 AI 번호 생성기")
    parser.add_argument(
        "--model", "-m",
        choices=["transformer", "lstm", "gpt"],
        default="transformer",
        help="모델 타입 선택 (기본: transformer)"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="생성할 번호 세트 수 (기본: 5)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="사용 가능한 모델 목록"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("사용 가능한 모델:")
        print("  - transformer: Transformer 기반 모델 (기본)")
        print("  - lstm: LSTM 기반 모델 (미구현)")
        print("  - gpt: GPT 기반 모델 (미구현)")
        return
    
    if args.model == "transformer":
        from models_ai.src.transformer.generate import main as transformer_main
        transformer_main()
    elif args.model == "lstm":
        print("❌ LSTM 모델은 아직 구현되지 않았습니다.")
    elif args.model == "gpt":
        print("❌ GPT 모델은 아직 구현되지 않았습니다.")


if __name__ == "__main__":
    main()
