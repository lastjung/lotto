"""
Base collector interface for all lottery data collectors.
각 복권별 수집기는 이 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
import json


from dataclasses import dataclass

@dataclass
class Draw:
    draw_no: int
    draw_date: str
    numbers: list[int]
    bonus: int | None = None

class BaseLotteryCollector(ABC):
    """복권 데이터 수집기 기본 클래스"""
    
    def __init__(self, lottery_id: str, config: dict):
        self.lottery_id = lottery_id
        self.config = config
        self.name = config.get("name", lottery_id)
        self.data_file = Path(config.get("data_file", f"data/{lottery_id}/draws.json"))
    
    @abstractmethod
    def get_latest_draw_no(self) -> int:
        """가장 최근 회차 번호를 반환합니다."""
        pass
    
    @abstractmethod
    def fetch_draw(self, draw_no: int) -> dict | None:
        """특정 회차 데이터를 가져옵니다."""
        pass
    
    def load_existing_data(self) -> dict:
        """기존 저장된 데이터를 로드합니다."""
        if self.data_file.exists():
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"draws": [], "updated_at": None}
    
    def save_data(self, data: dict):
        """데이터를 JSON 파일로 저장합니다."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        data["updated_at"] = datetime.now().isoformat()
        data["lottery_id"] = self.lottery_id
        data["lottery_name"] = self.name
        
        # Draw 객체가 있으면 dict로 변환
        draws = data.get("draws", [])
        if draws and hasattr(draws[0], "__dict__"):
            from dataclasses import asdict
            data["draws"] = [asdict(d) for d in draws]
            
        data["total_draws"] = len(data.get("draws", []))
        
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_draws(self, draws: list):
        """회차 목록만 받아서 저장합니다."""
        data = self.load_existing_data()
        data["draws"] = draws
        self.save_data(data)
    
    def get_last_saved_draw_no(self) -> int:
        """마지막으로 저장된 회차 번호를 반환합니다."""
        data = self.load_existing_data()
        draws = data.get("draws", [])
        if draws:
            return max(d.get("draw_no", 0) for d in draws)
        return 0
    
    def collect(self, start: int = None, end: int = None, update_only: bool = True) -> list:
        """
        데이터를 수집합니다.
        
        Args:
            start: 시작 회차 (None이면 1)
            end: 끝 회차 (None이면 최신)
            update_only: True면 누락된 회차만 채우기, False면 전체 다시 받아오기
        """
        from tqdm import tqdm
        import time
        
        # 최신 회차 확인
        latest_on_server = self.get_latest_draw_no()
        
        if end is None:
            end = latest_on_server
        
        if start is None:
            start = 1
            
        print(f"[{self.name}] 수집 범위 확인: {start}회 ~ {end}회")
        
        # 기존 데이터 로드
        existing_data = self.load_existing_data()
        existing_draws = existing_data.get("draws", [])
        existing_map = {d["draw_no"]: d for d in existing_draws}
        
        # 수집 대상 번호 목록 생성
        to_collect = []
        for draw_no in range(start, end + 1):
            if not update_only or draw_no not in existing_map:
                to_collect.append(draw_no)
        
        if not to_collect:
            print(f"[{self.name}] 이미 모든 데이터가 최신 상태입니다. (최신: {latest_on_server}회)")
            return []
        
        print(f"[{self.name}] 총 {len(to_collect)}개 회차 수집 시작...")
        
        # 새 데이터 수집
        new_draws_added = 0
        for draw_no in tqdm(to_collect, desc=f"수집 중 ({self.lottery_id})"):
            result = self.fetch_draw(draw_no)
            if result:
                existing_map[draw_no] = result
                new_draws_added += 1
                time.sleep(0.05)  # API 부하 방지
        
        # 정렬 후 저장
        updated_draws = sorted(existing_map.values(), key=lambda x: x["draw_no"])
        existing_data["draws"] = updated_draws
        self.save_data(existing_data)
        
        print(f"[{self.name}] ✅ {new_draws_added}개 회차 업데이트 완료 (총 {len(updated_draws)}회차)")
        
        return updated_draws # 전체 리스트 반환
