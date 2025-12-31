"""
일본 로또 6 데이터 수집기
데이터 출처: GitHub (tank1159jhs/jp-lottery-api)
※ 현재 최근 데이터만 수집 가능. 과거 전체 데이터는 미확보 상태입니다.
"""

import json
import requests
from datetime import datetime
from pathlib import Path

from collectors.base import BaseLotteryCollector, Draw


class JapanLoto6Collector(BaseLotteryCollector):
    """일본 Loto 6 데이터 수집기"""
    
    LOTTERY_ID = "japan_loto6"
    LOTTERY_NAME = "Japan Loto 6"
    
    # GitHub API endpoints
    GITHUB_API = "https://api.github.com/repos/tank1159jhs/jp-lottery-api/contents/data/loto6"
    GITHUB_RAW = "https://raw.githubusercontent.com/tank1159jhs/jp-lottery-api/main/data/loto6"
    
    def fetch_draw(self, draw_no: int) -> Draw | None:
        """특정 회차 데이터 조회"""
        try:
            url = f"{self.GITHUB_RAW}/{draw_no}.json"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            return Draw(
                draw_no=data["round"],
                draw_date=data["date"],
                numbers=sorted(data["numbers"]),
                bonus=data["bonus"]
            )
        except Exception:
            return None
    
    def fetch_all_data(self) -> list[Draw]:
        """GitHub에서 모든 데이터 수집"""
        try:
            # Get file list from GitHub
            response = requests.get(self.GITHUB_API, timeout=30)
            if response.status_code != 200:
                print(f"⚠️ GitHub API 호출 실패: {response.status_code}")
                return []
            
            files = response.json()
            print(f"[{self.LOTTERY_NAME}] {len(files)}개 파일 발견")
            
            draws = []
            for f in files:
                if not f['name'].endswith('.json'):
                    continue
                
                try:
                    url = f"{self.GITHUB_RAW}/{f['name']}"
                    data = requests.get(url, timeout=10).json()
                    
                    draws.append(Draw(
                        draw_no=data["round"],
                        draw_date=data["date"],
                        numbers=sorted(data["numbers"]),
                        bonus=data["bonus"]
                    ))
                except Exception:
                    continue
            
            # Sort by draw number
            draws.sort(key=lambda x: x.draw_no)
            return draws
            
        except Exception as e:
            print(f"⚠️ 데이터 수집 실패: {e}")
            return []
    
    def collect(self, update_only: bool = True) -> list[Draw]:
        """데이터 수집 (GitHub에서 전체 동기화)"""
        print(f"[{self.LOTTERY_NAME}] GitHub에서 데이터 다운로드 중...")
        
        draws = self.fetch_all_data()
        
        if draws:
            self.save_draws(draws)
            print(f"[{self.LOTTERY_NAME}] ✅ {len(draws)}개 회차 저장 완료")
        else:
            print(f"[{self.LOTTERY_NAME}] ❌ 데이터를 찾을 수 없습니다")
        
        return draws
