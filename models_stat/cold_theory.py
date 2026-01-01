"""
Cold Theory Model - 콜드 이론 (연체 번호 분석)

"오래 안 나온 번호는 나올 때가 됐다"는 역발상 이론:
- 오래 연체된 번호 우선 선택
- 최근 출현하지 않은 번호에 가중치
"""
import random
from collections import defaultdict
from typing import List, Dict


class ColdTheoryModel:
    """콜드 이론 기반 번호 생성 모델"""
    
    def __init__(self, draws: List[Dict], ball_count: int = 6, max_number: int = 45):
        self.draws = draws
        self.ball_count = ball_count
        self.max_number = max_number
        
        # 분석 결과
        self.overdue_counts = {}  # 번호별 연체 회차
        self.last_appearance = {}  # 번호별 마지막 출현 회차
        
        if draws:
            self._analyze()
    
    def _analyze(self):
        """연체 분석 수행"""
        total_draws = len(self.draws)
        
        # 각 번호의 마지막 출현 위치 찾기
        for num in range(1, self.max_number + 1):
            self.last_appearance[num] = 0  # 한번도 안 나옴
            
        for idx, draw in enumerate(self.draws):
            for num in draw.get("numbers", []):
                self.last_appearance[num] = idx + 1  # 1-indexed
        
        # 연체 회차 계산 (현재 회차 - 마지막 출현)
        for num in range(1, self.max_number + 1):
            if self.last_appearance[num] == 0:
                self.overdue_counts[num] = total_draws  # 한번도 안 나온 경우
            else:
                self.overdue_counts[num] = total_draws - self.last_appearance[num]
    
    def get_most_overdue(self, count: int = 10) -> List[tuple]:
        """가장 오래 안 나온 번호들 반환"""
        sorted_nums = sorted(self.overdue_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_nums[:count]
    
    def generate(self, count: int = 5, strategy: str = "weighted") -> List[Dict]:
        """
        콜드 이론 기반 번호 생성
        
        Args:
            count: 생성할 조합 수
            strategy:
                - "pure_cold": 가장 오래된 번호만 선택
                - "weighted": 연체 기간에 비례한 가중치
                - "mixed": 콜드 + 일부 랜덤
        """
        results = []
        seen = set()
        max_attempts = count * 50
        attempts = 0
        
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            
            if strategy == "pure_cold":
                numbers = self._generate_pure_cold()
            elif strategy == "weighted":
                numbers = self._generate_weighted()
            else:  # mixed
                numbers = self._generate_mixed()
            
            numbers_tuple = tuple(sorted(numbers))
            
            if (len(set(numbers)) == self.ball_count and 
                numbers_tuple not in seen and
                all(1 <= n <= self.max_number for n in numbers)):
                
                seen.add(numbers_tuple)
                analysis = self._analyze_numbers(list(numbers_tuple))
                results.append({
                    "numbers": list(numbers_tuple),
                    "analysis": analysis
                })
        
        return results
    
    def _generate_pure_cold(self) -> List[int]:
        """가장 연체된 번호들만 선택"""
        sorted_nums = sorted(range(1, self.max_number + 1), 
                           key=lambda x: self.overdue_counts.get(x, 0), 
                           reverse=True)
        
        # 상위 연체 번호 중에서 랜덤 선택 (약간의 변동성)
        top_cold = sorted_nums[:self.ball_count * 3]
        return random.sample(top_cold, self.ball_count)
    
    def _generate_weighted(self) -> List[int]:
        """연체 기간에 비례한 가중치로 선택"""
        weights = [self.overdue_counts.get(n, 1) + 1 for n in range(1, self.max_number + 1)]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        
        numbers = []
        available = list(range(1, self.max_number + 1))
        
        for _ in range(self.ball_count):
            probs = [probabilities[n-1] if n in available else 0 
                    for n in range(1, self.max_number + 1)]
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p / total_prob for p in probs]
            
            chosen = random.choices(range(1, self.max_number + 1), weights=probs, k=1)[0]
            numbers.append(chosen)
            available.remove(chosen)
        
        return numbers
    
    def _generate_mixed(self) -> List[int]:
        """콜드 + 랜덤 혼합"""
        # 절반은 콜드, 절반은 랜덤
        cold_count = self.ball_count // 2
        random_count = self.ball_count - cold_count
        
        sorted_nums = sorted(range(1, self.max_number + 1),
                           key=lambda x: self.overdue_counts.get(x, 0),
                           reverse=True)
        
        cold_picks = random.sample(sorted_nums[:cold_count * 3], cold_count)
        remaining = [n for n in range(1, self.max_number + 1) if n not in cold_picks]
        random_picks = random.sample(remaining, random_count)
        
        return cold_picks + random_picks
    
    def _analyze_numbers(self, numbers: List[int]) -> Dict:
        """생성된 번호 분석"""
        overdue_values = [self.overdue_counts.get(n, 0) for n in numbers]
        avg_overdue = sum(overdue_values) / len(overdue_values)
        max_overdue = max(overdue_values)
        
        return {
            "sum": sum(numbers),
            "avg_overdue": round(avg_overdue, 1),
            "max_overdue": max_overdue,
            "overdue_rating": self._get_overdue_rating(avg_overdue)
        }
    
    def _get_overdue_rating(self, avg: float) -> str:
        """연체 등급"""
        if avg >= 30:
            return "매우 연체 (강추)"
        elif avg >= 15:
            return "연체 (추천)"
        else:
            return "보통"


def create_model(draws: List[Dict], ball_count: int = 6, max_number: int = 45) -> ColdTheoryModel:
    """모델 생성 헬퍼"""
    return ColdTheoryModel(draws, ball_count, max_number)
