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
    
    MAX_RETRIES = 3
    DB_URL = "https://raw.githubusercontent.com/CorentinLeGuen/lotto-6-49-api/main/db.sqlite3"
    
    def ensure_db_exists(self):
        """SQLite DB 파일이 없으면 자동 다운로드"""
        if self.DB_PATH.exists():
            return

        print(f"[{self.LOTTERY_NAME}] DB 파일 다운로드 중... (Source: GitHub)")
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        import urllib.request
        try:
            urllib.request.urlretrieve(self.DB_URL, self.DB_PATH)
            print(f"[{self.LOTTERY_NAME}] ✅ DB 다운로드 완료")
        except Exception as e:
            print(f"[{self.LOTTERY_NAME}] ❌ DB 다운로드 실패: {e}")

    def get_latest_draw_no(self) -> int:
        """최신 회차 번호 조회"""
        self.ensure_db_exists()
        
        if not self.DB_PATH.exists():
            return 0
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM draw')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def fetch_draw(self, draw_no: int) -> Draw | None:
        """특정 회차 데이터 조회 (SQLite에서)"""
        self.ensure_db_exists()
        
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
        self.ensure_db_exists()
        
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
    
    WCLC_URL = "https://www.wclc.com/winning-numbers/lotto-649-extra.htm"

    def scrape_wclc(self, year: int) -> list[Draw]:
        """WCLC 웹사이트에서 특정 연도의 데이터 스크래핑"""
        print(f"[{self.LOTTERY_NAME}] {year}년 데이터 스크래핑 중... ({self.WCLC_URL})")
        
        import urllib.request
        import re
        
        url = f"{self.WCLC_URL}?year={year}"
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
        except Exception as e:
            print(f"[{self.LOTTERY_NAME}] ❌ 스크래핑 실패 ({year}): {e}")
            return []

        # WCLC HTML 구조 파싱
        draws = []
        months = {
            "January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06",
            "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"
        }

        # 1. Split by 'pastWinNumGroupDivider' div which contains each draw
        # <div class="pastWinNumGroup pastWinNumGroupDivider">
        groups = html.split('class="pastWinNumGroup pastWinNumGroupDivider"')
        
        # The date is usually in the PREVIOUS text block or header.
        # WCLC structure:
        # <div ...> <h4>Date</h4> </div> <div class="pastWinNumGroup ..."> ... </div>
        
        # Better strategy: Find all H4 headers (dates) and their following sibling divs (numbers).
        # Since logic is tricky with regex/split, let's look at the H4 and subsequent content.
        
        # Split by <h4> to get chunks starting with date
        parts = html.split('<h4>')
        
        for part in parts[1:]: # Skip preamble
            # Part starts with: "\n\t\t\t\tWednesday, December 31, 2025\n\t\t\t\t</h4>..."
            
            # 1. Parse Date
            if '</h4>' not in part: continue
            header_text, content = part.split('</h4>', 1)
            
            # Clean up whitespace
            header_text = header_text.strip()
            # Regex: Wednesday, December 31, 2025
            date_match = re.search(r'\w+,\s+(\w+)\s+(\d+),\s+(\d+)', header_text)
            if not date_match: continue
            
            month_str, day, year_str = date_match.groups()
            
            # Check year match (WCLC page sometimes shows prev year in late Dec or generic list)
            if int(year_str) != year: continue
            
            month = months.get(month_str)
            if not month: continue
            
            draw_date = f"{year_str}-{month}-{day.zfill(2)}"
            
            # 2. Parse Numbers
            # Look for CLASSIC DRAW only, ignore "GOLD BALL DRAW" if present in same block?
            # The structure for numbers is in <ul class="pastWinNumbers">
            
            if 'class="pastWinNumbers"' not in content: continue
            
            # Extract the UL block
            ul_block = content.split('class="pastWinNumbers"')[1].split('</ul>')[0]
            
            # Main numbers: <li class="pastWinNumber">10</li>
            main_nums = re.findall(r'class="pastWinNumber">(\d+)<', ul_block)
            main_nums = [int(n) for n in main_nums]
            
            # Bonus number: <li class="pastWinNumberBonus">...</span>28</li>
            # Regex matches number after ANY tag closure or directly
            # Handle: <span ...>Bonus</span>28
            bonus_match = re.search(r'class="pastWinNumberBonus".*?>(\d+)<', ul_block, re.DOTALL)
            # If not found, look for text after span
            if not bonus_match:
                 bonus_match = re.search(r'class="pastWinNumberBonus".*?</span>\s*(\d+)', ul_block, re.DOTALL)
            
            bonus = int(bonus_match.group(1)) if bonus_match else None
            
            if len(main_nums) >= 6:
                # WCLC 6/49 has 6 main numbers.
                # If parsed correctly, we trust it.
                draws.append(Draw(
                    draw_no=0,
                    draw_date=draw_date,
                    numbers=sorted(main_nums[:6]),
                    bonus=bonus
                ))

        print(f"[{self.LOTTERY_NAME}] {len(draws)}개 회차 스크래핑 성공 ({year})")
        return draws

    def collect(self, update_only: bool = True) -> list[Draw]:
        """하이브리드 데이터 수집 (SQLite + Web Scraping)"""
        print(f"[{self.LOTTERY_NAME}] 하이브리드 수집 시작 (SQLite + WCLC Scraping)...")
        
        # 1. SQLite 데이터 로드 (Fast, ~2023)
        sqlite_draws = self.fetch_all_data() # ensure_db_exists 내부 호출됨
        if not sqlite_draws:
            print(f"[{self.LOTTERY_NAME}] SQLite 로드 실패, 스크래핑만 시도합니다.")
        
        # 2. 최신 데이터 확인
        latest_date_str = "1982-01-01"
        if sqlite_draws:
            # draw_date 기준 정렬
            sqlite_draws.sort(key=lambda x: x.draw_date)
            latest_date_str = sqlite_draws[-1].draw_date
            
        latest_year = int(latest_date_str.split('-')[0])
        current_year = datetime.now().year
        
        print(f"[{self.LOTTERY_NAME}] SQLite 최신 데이터: {latest_date_str} (스크래핑 대상: {latest_year}~{current_year})")
        
        # 3. 빈 연도 스크래핑 (최신 연도 포함)
        scraped_draws = []
        for year in range(latest_year, current_year + 1):
            scraped_draws.extend(self.scrape_wclc(year))
            
        # 4. 병합 (중복 제거)
        # 날짜를 키로 사용하여 중복 제거 (스크래핑이 더 최신/정확할 수 있으므로 덮어쓰기 고려? 아니면 SQLite 우선?)
        # SQLite: Verified historical data. Scraping: Live data.
        # 날짜가 같으면 SQLite 우선 (보너스 번호 처리 등 안정성)
        
        draw_map = {d.draw_date: d for d in sqlite_draws}
        
        new_count = 0
        for draw in scraped_draws:
            if draw.draw_date not in draw_map:
                draw_map[draw.draw_date] = draw
                new_count += 1
                
        # 5. 최종 리스트 생성 및 회차 번호(draw_no) 재할당
        # Canada 6/49는 1982-06-12 부터 시작. 
        # 중간에 누락이 없다면 날짜순 정렬 후 인덱스+1이 회차 번호가 됨.
        # (정확한 회차 번호를 알기 어려우므로 이 방식 사용)
        
        final_draws = sorted(draw_map.values(), key=lambda x: x.draw_date)
        
        # 회차 재할당 (SQLite에 있는 draw_no가 정확하다면 유지하고, 뒤이어 붙이기)
        # 하지만 스크래핑 데이터는 draw_no가 0임.
        # 전체를 재할당하는 것이 안전 (단, 1회차 날짜가 맞다면)
        
        for i, d in enumerate(final_draws, 1):
            d.draw_no = i
            
        # 6. 저장
        self.save_draws(final_draws)
        
        print(f"[{self.LOTTERY_NAME}] ✅ 총 {len(final_draws)}회차 저장 완료 (신규/스크래핑: {new_count}건)")
        return final_draws
