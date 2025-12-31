"""
미국 메가 밀리언즈 데이터 수집기
데이터 출처: NY Data.gov (https://data.ny.gov/resource/5xaw-6ayf.json)
"""

import json
import requests
from datetime import datetime
from pathlib import Path

from collectors.base import BaseLotteryCollector, Draw


class USAMegaMillionsCollector(BaseLotteryCollector):
    """미국 Mega Millions 데이터 수집기"""
    
    LOTTERY_ID = "usa_megamillions"
    LOTTERY_NAME = "Mega Millions"
    
    # NY Data.gov API (Mega Millions)
    API_URL = "https://data.ny.gov/resource/5xaw-6ayf.json"
    
    def fetch_draw(self, draw_no: int) -> Draw | None:
        """특정 회차는 지원하지 않음 (API가 날짜 기반)"""
        return None
    
    def fetch_all_data(self) -> list[Draw]:
        """NY Data.gov에서 전체 데이터 수집"""
        try:
            # 전체 데이터 요청
            url = f"{self.API_URL}?$limit=50000&$order=draw_date ASC"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; LottoCollector/1.0)"
            }
            
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            print(f"[{self.LOTTERY_NAME}] {len(data)}개 데이터 수집됨")
            
            draws = []
            for i, item in enumerate(data, 1):
                try:
                    # 날짜 파싱
                    draw_date = item["draw_date"][:10]  # "2025-12-26T00:00:00.000" -> "2025-12-26"
                    
                    # 번호 파싱: "09 19 31 63 64" -> [9, 19, 31, 63, 64]
                    numbers = [int(n) for n in item["winning_numbers"].split()]
                    
                    # 메가볼
                    mega_ball = int(item["mega_ball"])
                    
                    draws.append(Draw(
                        draw_no=i,  # 순차 번호 부여
                        draw_date=draw_date,
                        numbers=sorted(numbers),
                        bonus=mega_ball
                    ))
                except Exception as e:
                    continue
            
            return draws
            
        except Exception as e:
            print(f"⚠️ 데이터 수집 실패: {e}")
            return []
    
    def collect(self, update_only: bool = True) -> list[Draw]:
        """데이터 수집 (항상 전체 갱신하여 순차 번호 일관성 유지)"""
        print(f"[{self.LOTTERY_NAME}] NY Data.gov에서 데이터 수집 중...")
        
        draws = self.fetch_all_data()
        
        if draws:
            self.save_draws(draws)
            print(f"[{self.LOTTERY_NAME}] ✅ {len(draws)}회차 저장 완료")
        else:
            print(f"[{self.LOTTERY_NAME}] ❌ 데이터를 찾을 수 없습니다")
        
        return draws
