"""
모든 등록된 복권 데이터를 업데이트하는 스크립트
설정 파일을 읽어서 각 복권별 수집기를 실행합니다.
"""

import json
import argparse
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors import get_collector, list_available_collectors


def load_config() -> dict:
    """로또 설정 파일 로드"""
    config_path = Path(__file__).parent.parent / "config" / "lotteries.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_lottery(lottery_id: str, config: dict, full: bool = False):
    """단일 복권 데이터 업데이트"""
    try:
        collector = get_collector(lottery_id, config)
        collector.collect(update_only=not full)
    except ValueError as e:
        print(f"⚠️  {lottery_id}: {e} (수집기 미구현)")
    except Exception as e:
        print(f"❌ {lottery_id} 수집 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="복권 데이터 업데이트")
    parser.add_argument(
        "--lottery", "-l",
        type=str,
        default=None,
        help="특정 복권만 업데이트 (예: korea_645)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="전체 데이터 재수집 (기본: 새 데이터만)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 수집기 목록 표시"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("사용 가능한 수집기:")
        for name in list_available_collectors():
            print(f"  - {name}")
        return
    
    config = load_config()
    
    print("=" * 50)
    print("🎱 복권 데이터 업데이트")
    print("=" * 50)
    
    if args.lottery:
        # 특정 복권만
        if args.lottery not in config:
            print(f"❌ 알 수 없는 복권: {args.lottery}")
            print(f"   사용 가능: {list(config.keys())}")
            return
        update_lottery(args.lottery, config[args.lottery], args.full)
    else:
        # 모든 복권
        for lottery_id, lottery_config in config.items():
            print(f"\n--- {lottery_config['name']} ---")
            update_lottery(lottery_id, lottery_config, args.full)
    
    print("\n" + "=" * 50)
    print("✅ 완료")


if __name__ == "__main__":
    main()
