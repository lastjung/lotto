"""
누락된 복권 데이터를 수동으로 주입하거나 업데이트하는 스크립트
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors import get_collector

def fill_lottery_data(lottery_id, new_draws_data):
    """
    특정 복권의 JSON 파일에 데이터를 주입합니다.
    new_draws_data: list of dicts with {draw_no, draw_date, numbers, bonus}
    """
    config_path = Path(__file__).parent.parent / "config" / "lotteries.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if lottery_id not in config:
        print(f"❌ 알 수 없는 복권: {lottery_id}")
        return

    collector = get_collector(lottery_id, config[lottery_id])
    existing_data = collector.load_existing_data()
    existing_draws = existing_data.get("draws", [])
    
    # Map for easy replacement
    draw_map = {d["draw_no"]: d for d in existing_draws}
    
    added_count = 0
    updated_count = 0
    
    for draw in new_draws_data:
        draw_no = draw["draw_no"]
        if draw_no in draw_map:
            draw_map[draw_no].update(draw)
            updated_count += 1
        else:
            draw_map[draw_no] = draw
            added_count += 1
            
    # Sort and save
    sorted_draws = sorted(draw_map.values(), key=lambda x: x["draw_no"])
    existing_data["draws"] = sorted_draws
    collector.save_data(existing_data)
    
    print(f"[{lottery_id}] ✅ 추가: {added_count}건, 업데이트: {updated_count}건 (총 {len(sorted_draws)}건)")

if __name__ == "__main__":
    # 예시 데이터 (캐나다 2025/2024 일부)
    # 실제 데이터는 스크래핑 결과를 바탕으로 채워야 함
    canada_scraped = [
        {"draw_no": 4410, "draw_date": "2025-12-27", "numbers": [26, 27, 30, 38, 47, 48], "bonus": 43},
        {"draw_no": 4409, "draw_date": "2025-12-24", "numbers": [10, 23, 30, 42, 44, 46], "bonus": 14},
        {"draw_no": 4408, "draw_date": "2025-12-20", "numbers": [1, 5, 20, 29, 36, 47], "bonus": 5},
        {"draw_no": 4407, "draw_date": "2025-12-17", "numbers": [2, 19, 20, 22, 42, 47], "bonus": 35},
        {"draw_no": 4406, "draw_date": "2025-12-13", "numbers": [2, 8, 23, 28, 29, 36], "bonus": 13},
        {"draw_no": 4405, "draw_date": "2025-12-10", "numbers": [8, 14, 21, 25, 43, 49], "bonus": 4},
        {"draw_no": 4404, "draw_date": "2025-12-06", "numbers": [1, 23, 25, 30, 43, 45], "bonus": 46},
        {"draw_no": 4403, "draw_date": "2025-12-03", "numbers": [7, 10, 16, 29, 39, 44], "bonus": 40},
    ]
    
    fill_lottery_data("canada_649", canada_scraped)
    print("스크립트가 준비되었습니다. 주입할 데이터를 배열에 넣고 호출하세요.")
