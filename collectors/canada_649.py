"""
캐나다 로또 6/49 데이터 수집기
데이터 출처: GitHub (CorentinLeGuen/lotto-6-49-api) SQLite 데이터베이스
※ 이 수집기는 오프라인 데이터를 사용하며, 공개 API가 없어 자동 업데이트가 제한됩니다.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

from collectors.base import BaseLotteryCollector, Draw


class CanadaLottoCollector(BaseLotteryCollector):
    """캐나다 Lotto 6/49 데이터 수집기"""
    
    LOTTERY_ID = "canada_649"
    LOTTERY_NAME = "Canada Lotto 6/49"
    
    # SQLite 데이터베이스 경로
    DB_PATH = Path(__file__).parent.parent / "data" / "canada_649" / "db.sqlite3"
    
    def fetch_draw(self, draw_no: int) -> Draw | None:
        """특정 회차 데이터 조회 (SQLite에서)"""
        if not self.DB_PATH.exists():
            return None
            
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # draw_no로 직접 조회 (offset 사용)
        cursor.execute('''
            SELECT draw_date, number_1, number_2, number_3, number_4, number_5, number_6, number_c 
            FROM draw 
            ORDER BY draw_date ASC
            LIMIT 1 OFFSET ?
        ''', (draw_no - 1,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        draw_date, n1, n2, n3, n4, n5, n6, bonus = row
        return Draw(
            draw_no=draw_no,
            draw_date=draw_date,
            numbers=sorted([n1, n2, n3, n4, n5, n6]),
            bonus=bonus
        )
    
    def fetch_all_data(self) -> list[Draw]:
        """전체 데이터를 SQLite에서 읽어서 반환"""
        if not self.DB_PATH.exists():
            print(f"⚠️ SQLite 파일이 없습니다: {self.DB_PATH}")
            return []
        
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT draw_date, number_1, number_2, number_3, number_4, number_5, number_6, number_c 
            FROM draw 
            ORDER BY draw_date ASC
        ''')
        
        draws = []
        for i, row in enumerate(cursor.fetchall(), 1):
            draw_date, n1, n2, n3, n4, n5, n6, bonus = row
            draws.append(Draw(
                draw_no=i,
                draw_date=draw_date,
                numbers=sorted([n1, n2, n3, n4, n5, n6]),
                bonus=bonus
            ))
        
        conn.close()
        return draws
    
    def collect(self, update_only: bool = True) -> list[Draw]:
        """데이터 수집 (SQLite에서 JSON으로 변환)"""
        print(f"[{self.LOTTERY_NAME}] SQLite 데이터베이스에서 데이터 로드 중...")
        
        draws = self.fetch_all_data()
        
        if draws:
            self.save_draws(draws)
            print(f"[{self.LOTTERY_NAME}] ✅ {len(draws)}회차 데이터 저장 완료")
        else:
            print(f"[{self.LOTTERY_NAME}] ❌ 데이터를 찾을 수 없습니다")
        
        return draws
