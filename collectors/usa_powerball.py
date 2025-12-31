"""
미국 파워볼(USA Powerball) 데이터 수집기
출처: 
- NY Data.gov (https://data.ny.gov/)
- NC Education Lottery (https://nclottery.com/)
공개된 CSV/JSON 데이터를 사용합니다. (NY 소스 불안정 시 NC 소스 활용)
"""

import requests
import time
from datetime import datetime
from .base import BaseLotteryCollector

class USAPowerballCollector(BaseLotteryCollector):
    """
    미국 파워볼 수집기 (NY Data.gov JSON Source)
    
    참고: 현재는 당첨 번호와 Multiplier 정보만 수집합니다.
    1등 총상금(Jackpot) 및 당첨자 수 정보는 공공 CSV 소스의 제약으로 인해 
    제외되어 있으며, 필요 시 별도의 전용 API 연동이 필요합니다.
    """
    
    # NY Data.gov JSON API (더 안정적임)
    API_URL = "https://data.ny.gov/resource/d6yy-54nr.json?$limit=5000"
    
    def get_latest_draw_no(self) -> int:
        return 99999999 # 항상 최신 데이터 확인

    def fetch_all_data(self) -> list:
        """NY Data.gov JSON 데이터를 다운로드하여 가공합니다."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[{self.name}] NY Data.gov에서 JSON 데이터 다운로드 중... (시도 {attempt+1}/{max_retries})")
                response = requests.get(self.API_URL, timeout=60)
                response.raise_for_status()
                
                raw_data = response.json()
                draws = []
                
                for item in raw_data:
                    try:
                        # "2024-12-28T00:00:00.000" 형식
                        date_str = item["draw_date"].split("T")[0]
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        # "11 19 34 48 53 21" 형식
                        nums_str = item["winning_numbers"].split()
                        if len(nums_str) < 6:
                            continue
                            
                        main_numbers = sorted([int(n) for n in nums_str[:5]])
                        powerball = int(nums_str[5])
                        
                        draws.append({
                            "draw_date": date_str,
                            "numbers": main_numbers,
                            "bonus": powerball,
                            "multiplier": item.get("multiplier", "")
                        })
                    except (ValueError, KeyError, IndexError):
                        continue
                
                # 날짜순 정렬 후 순차적 회차 번호 부여 (1, 2, 3...)
                # 파워볼 공식 회차 번호가 데이터에 없으므로 우리 시스템 기준 시퀀스 사용
                draws.sort(key=lambda x: x["draw_date"])
                for i, draw in enumerate(draws):
                    draw["draw_no"] = i + 1
                    
                return draws
                
            except Exception as e:
                print(f"⚠️ 시도 {attempt+1} 실패: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return []

    def collect(self, start: int = None, end: int = None, update_only: bool = True) -> list:
        all_remote_draws = self.fetch_all_data()
        if not all_remote_draws:
            print(f"[{self.name}] 데이터 수집에 실패했습니다.")
            return []
            
        # 모든 데이터를 새로 덮어씁니다 (순차적 회차 번호 정합성 유지 위함)
        existing_data = self.load_existing_data()
        existing_data["draws"] = all_remote_draws
        self.save_data(existing_data)
        
        print(f"[{self.name}] ✅ 최신 데이터로 동기화 완료 (총 {len(all_remote_draws)}회차, 순차 번호 부여)")
        return all_remote_draws

    def fetch_draw(self, draw_no: int) -> dict | None:
        return None


def create_collector(config: dict) -> USAPowerballCollector:
    return USAPowerballCollector("usa_powerball", config)
