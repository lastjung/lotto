"""
모든 등록된 복권 데이터를 업데이트하는 스크립트
설정 파일을 읽어서 각 복권별 수집기를 실행합니다.
"""

import json
import argparse
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# 로또 프로젝트 루트 path 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 로드
load_dotenv()

from collectors import get_collector, list_available_collectors

# Supabase 클라이언트 설정
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # 쓰기 권한은 Service Role 키 권장
    
    # Service Role 키가 없으면 Anon Key 시도 (단, RLS 설정에 따라 실패할 수 있음)
    if not SUPABASE_KEY:
        SUPABASE_KEY = os.environ.get("VITE_SUPABASE_ANON_KEY")
        
    supabase: Client = None
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except ImportError:
    supabase = None
    print("⚠️ 'supabase' 패키지가 설치되지 않았습니다. DB 업로드가 건너뛰어집니다.")


def load_config() -> dict:
    """로또 설정 파일 로드"""
    config_path = Path(__file__).parent.parent / "config" / "lotteries.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_to_supabase(lottery_id: str, draws: list):
    """데이터를 Supabase에 업로드 (Batch Upsert)"""
    if not supabase:
        print(f"⚠️ Supabase 클라이언트가 초기화되지 않았습니다. (키 확인 필요)")
        return

    if not draws:
        return

    print(f"[{lottery_id}] Supabase 업로드 시작 ({len(draws)}건)...")
    
    # 데이터 변환 (Draw 객체 -> dict)
    # lotto_history 테이블 스키마 가정:
    # lottery_id, draw_no, draw_date, numbers(int array), bonus(int)
    
    # 중복 제거 (draw_no 기준)
    unique_records = {}
    for draw in draws:
        if isinstance(draw, dict):
            d_no = draw.get("draw_no")
            d_date = draw.get("draw_date")
            nums = draw.get("numbers")
            bonus = draw.get("bonus")
        else:
            d_no = draw.draw_no
            d_date = draw.draw_date
            nums = draw.numbers
            bonus = getattr(draw, "bonus", None)

        # 딕셔너리에 저장 (덮어쓰기로 최신 데이터 유지, 또는 첫번째 유지)
        # 여기서는 단순히 draw_no를 키로 사용하여 중복 방지
        unique_records[d_no] = {
            "lottery_id": lottery_id,
            "draw_no": d_no,
            "draw_date": d_date,
            "numbers": nums,
            "bonus": bonus
        }
    
    records = list(unique_records.values())
    
    # 500건씩 배치 처리
    batch_size = 500
    total = len(records)
    
    try:
        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]
            # upsert: conflict 발생 시(lottery_id, draw_no 복합키) 업데이트
            # 주의: Supabase 테이블에 해당 제약조건(Unique Key)이 있어야 함.
            # 보통 (lottery_id, draw_no)가 PK 또는 Unique.
            
            response = supabase.table("lotto_history").upsert(batch, on_conflict="lottery_id, draw_no").execute()
            # print(f"  - Batch {i//batch_size + 1}: {len(batch)}건 업로드 완료")
            
        print(f"[{lottery_id}] ✅ Supabase 동기화 완료 (총 {total}건)")
        
    except Exception as e:
        print(f"[{lottery_id}] ❌ Supabase 업로드 실패: {e}")


def update_lottery(lottery_id: str, config: dict, full: bool = False):
    """단일 복권 데이터 업데이트 및 DB 동기화"""
    try:
        collector = get_collector(lottery_id, config)
        # 1. 수집 (로컬 저장)
        draws = collector.collect(update_only=not full)
        
        # 2. DB 업로드 (옵션)
        if draws:
            upload_to_supabase(lottery_id, draws)
            
    except ValueError as e:
        print(f"⚠️  {lottery_id}: {e} (수집기 미구현)")
    except Exception as e:
        print(f"❌ {lottery_id} 수집 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="복권 데이터 업데이트 및 DB 동기화")
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
    print("🎱 복권 데이터 업데이트 & DB Sync")
    print(f"📡 Supabase 연결: {'✅' if supabase else '❌ (키 없음)'}")
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
