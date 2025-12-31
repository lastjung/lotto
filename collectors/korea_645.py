"""
한국 로또 6/45 데이터 수집기
출처: 동행복권 (https://www.dhlottery.co.kr/)
동행복권 공식 검색 API를 사용합니다.
"""

import requests
from datetime import datetime
from .base import BaseCollector


class Korea645Collector(BaseCollector):
    """한국 로또 6/45 수집기"""
    
    API_URL = "https://www.dhlottery.co.kr/common.do"
    
    def get_latest_draw_no(self) -> int:
        """최신 회차 추정 및 확인"""
        # 첫 추첨일 기준으로 대략 계산
        lotto_start = datetime(2002, 12, 7)
        weeks_passed = (datetime.now() - lotto_start).days // 7
        
        # 위에서 아래로 유효한 회차 찾기
        for draw_no in range(weeks_passed + 10, weeks_passed - 10, -1):
            if self.fetch_draw(draw_no):
                return draw_no
        
        return weeks_passed
    
    def fetch_draw(self, draw_no: int) -> dict | None:
        """특정 회차 데이터 가져오기"""
        params = {
            "method": "getLottoNumber",
            "drwNo": draw_no
        }
        
        try:
            response = requests.get(self.API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("returnValue") == "success":
                return {
                    "draw_no": data["drwNo"],
                    "draw_date": data["drwNoDate"],
                    "numbers": sorted([
                        data["drwtNo1"],
                        data["drwtNo2"],
                        data["drwtNo3"],
                        data["drwtNo4"],
                        data["drwtNo5"],
                        data["drwtNo6"]
                    ]),
                    "bonus": data["bnusNo"],
                    "total_sell_amount": data.get("totSellamnt"),
                    "first_prize_amount": data.get("firstWinamnt"),
                    "first_prize_winners": data.get("firstPrzwnerCo"),
                }
            return None
        except Exception as e:
            print(f"Error fetching draw {draw_no}: {e}")
            return None


# 편의를 위한 함수
def create_collector(config: dict) -> Korea645Collector:
    return Korea645Collector("korea_645", config)
