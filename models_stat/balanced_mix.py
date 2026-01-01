"""
Balanced Mix Model - 균형 혼합 모델

표준편차 최소화를 통한 균형 잡힌 번호 생성:
- 합계가 평균에 가깝게
- 홀짝 비율 균형
- 고저 비율 균형
"""
import random
from typing import List, Dict


class BalancedMixModel:
    """균형 혼합 기반 번호 생성 모델"""
    
    def __init__(self, draws: List[Dict], ball_count: int = 6, max_number: int = 45):
        self.draws = draws
        self.ball_count = ball_count
        self.max_number = max_number
        
        # 이상적인 통계값 계산
        self.ideal_sum = self._calculate_ideal_sum()
        self.ideal_odd_count = ball_count // 2  # 홀짝 균형
        self.midpoint = max_number // 2
        
    def _calculate_ideal_sum(self) -> float:
        """이상적인 합계 계산 (이론적 평균)"""
        # 이론적 합계: (1 + max_number) / 2 * ball_count
        return (1 + self.max_number) / 2 * self.ball_count
    
    def generate(self, count: int = 5, tolerance: float = 0.15) -> List[Dict]:
        """
        균형 잡힌 번호 생성
        
        Args:
            count: 생성할 조합 수
            tolerance: 허용 오차 (기본 15%)
        """
        results = []
        seen = set()
        max_attempts = count * 100
        attempts = 0
        
        # 허용 범위 계산
        sum_tolerance = self.ideal_sum * tolerance
        min_sum = int(self.ideal_sum - sum_tolerance)
        max_sum = int(self.ideal_sum + sum_tolerance)
        
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            numbers = self._generate_balanced()
            numbers_tuple = tuple(sorted(numbers))
            
            if numbers_tuple in seen:
                continue
            
            if len(set(numbers)) != self.ball_count:
                continue
            
            # 균형 검사
            total = sum(numbers)
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            low_count = sum(1 for n in numbers if n <= self.midpoint)
            
            # 합계 범위 체크
            if not (min_sum <= total <= max_sum):
                continue
            
            # 홀짝 균형 체크 (완전 균형 또는 ±1)
            if abs(odd_count - self.ideal_odd_count) > 1:
                continue
            
            # 고저 균형 체크
            if abs(low_count - self.ball_count // 2) > 1:
                continue
            
            seen.add(numbers_tuple)
            analysis = self._analyze_numbers(list(numbers_tuple))
            results.append({
                "numbers": list(numbers_tuple),
                "analysis": analysis
            })
        
        return results
    
    def _generate_balanced(self) -> List[int]:
        """균형 잡힌 번호 생성 시도"""
        numbers = []
        
        # 저역에서 절반, 고역에서 절반
        low_count = self.ball_count // 2
        high_count = self.ball_count - low_count
        
        low_range = list(range(1, self.midpoint + 1))
        high_range = list(range(self.midpoint + 1, self.max_number + 1))
        
        # 홀짝 균형을 고려하여 선택
        low_picks = random.sample(low_range, min(low_count, len(low_range)))
        high_picks = random.sample(high_range, min(high_count, len(high_range)))
        
        numbers = low_picks + high_picks
        
        # 부족하면 채우기
        while len(numbers) < self.ball_count:
            remaining = [n for n in range(1, self.max_number + 1) if n not in numbers]
            if remaining:
                numbers.append(random.choice(remaining))
        
        return numbers
    
    def _analyze_numbers(self, numbers: List[int]) -> Dict:
        """생성된 번호 분석"""
        total = sum(numbers)
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = self.ball_count - odd_count
        low_count = sum(1 for n in numbers if n <= self.midpoint)
        high_count = self.ball_count - low_count
        
        # 편차 계산
        sum_deviation = abs(total - self.ideal_sum) / self.ideal_sum * 100
        
        return {
            "sum": total,
            "odd_even": f"{odd_count}:{even_count}",
            "low_high": f"{low_count}:{high_count}",
            "deviation": round(sum_deviation, 1),
            "balance_rating": self._get_balance_rating(sum_deviation, odd_count, low_count)
        }
    
    def _get_balance_rating(self, dev: float, odd: int, low: int) -> str:
        """균형 등급"""
        half = self.ball_count // 2
        
        # 완벽한 균형
        if dev < 5 and abs(odd - half) <= 1 and abs(low - half) <= 1:
            return "완벽 균형 ⭐"
        elif dev < 10:
            return "우수 균형"
        elif dev < 15:
            return "양호"
        else:
            return "보통"


def create_model(draws: List[Dict], ball_count: int = 6, max_number: int = 45) -> BalancedMixModel:
    """모델 생성 헬퍼"""
    return BalancedMixModel(draws, ball_count, max_number)
