"""
Physics Bias Model - 물리적 편향 분석

로또 기계의 물리적 불완전성을 분석하여 번호 생성:
- 빈도 편향: 특정 번호가 더 자주 출현
- 위치 편향: 특정 순서에 특정 번호가 더 자주 출현
- 최근 트렌드: 핫/콜드 번호 분석
"""
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class PhysicsBiasModel:
    """물리적 편향 기반 번호 생성 모델"""
    
    def __init__(self, draws: List[Dict], ball_count: int = 6, max_number: int = 45):
        """
        Args:
            draws: 추첨 데이터 리스트 [{"numbers": [1,2,3,4,5,6], ...}, ...]
            ball_count: 공 개수 (6 or 5)
            max_number: 최대 번호 (45, 49, 69 등)
        """
        self.draws = draws
        self.ball_count = ball_count
        self.max_number = max_number
        
        # 분석 결과 저장
        self.frequency_bias = {}
        self.position_bias = defaultdict(dict)
        self.recent_hot = []
        self.recent_cold = []
        
        if draws:
            self._analyze()
    
    def _analyze(self):
        """전체 분석 수행"""
        self._analyze_frequency()
        self._analyze_position()
        self._analyze_recent_trend()
    
    def _analyze_frequency(self):
        """전체 빈도 분석 - 어떤 번호가 더 자주 나왔는지"""
        all_numbers = []
        for draw in self.draws:
            all_numbers.extend(draw.get("numbers", []))
        
        counter = Counter(all_numbers)
        total_draws = len(self.draws)
        
        # 각 번호의 출현 확률 계산 (1~max_number)
        for num in range(1, self.max_number + 1):
            count = counter.get(num, 0)
            # 기대 확률 대비 실제 확률
            expected = total_draws * self.ball_count / self.max_number
            self.frequency_bias[num] = count / expected if expected > 0 else 1.0
    
    def _analyze_position(self):
        """위치별 편향 분석 - 특정 순서에 특정 번호가 자주 나오는지"""
        for pos in range(self.ball_count):
            position_numbers = []
            for draw in self.draws:
                numbers = sorted(draw.get("numbers", []))
                if pos < len(numbers):
                    position_numbers.append(numbers[pos])
            
            counter = Counter(position_numbers)
            total = len(position_numbers)
            
            for num in range(1, self.max_number + 1):
                count = counter.get(num, 0)
                self.position_bias[pos][num] = count / total if total > 0 else 0
    
    def _analyze_recent_trend(self, recent_count: int = 30):
        """최근 트렌드 분석 - 핫/콜드 번호"""
        recent_draws = self.draws[-recent_count:] if len(self.draws) >= recent_count else self.draws
        
        recent_numbers = []
        for draw in recent_draws:
            recent_numbers.extend(draw.get("numbers", []))
        
        counter = Counter(recent_numbers)
        
        # 정렬하여 핫/콜드 분류
        sorted_nums = sorted(range(1, self.max_number + 1), 
                           key=lambda x: counter.get(x, 0), reverse=True)
        
        # 상위 1/4 = 핫, 하위 1/4 = 콜드
        quarter = self.max_number // 4
        self.recent_hot = sorted_nums[:quarter]
        self.recent_cold = sorted_nums[-quarter:]
    
    def generate(self, count: int = 5, strategy: str = "balanced") -> List[Dict]:
        """
        편향 기반 번호 생성
        
        Args:
            count: 생성할 조합 수
            strategy: 
                - "frequency": 빈도 편향만 사용
                - "position": 위치 편향만 사용
                - "hot": 핫 번호 우선
                - "balanced": 모든 전략 혼합 (기본)
        
        Returns:
            [{"numbers": [...], "analysis": {...}}, ...]
        """
        results = []
        seen = set()
        attempts = 0
        max_attempts = count * 50
        
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            
            if strategy == "frequency":
                numbers = self._generate_by_frequency()
            elif strategy == "position":
                numbers = self._generate_by_position()
            elif strategy == "hot":
                numbers = self._generate_hot_numbers()
            else:  # balanced
                numbers = self._generate_balanced()
            
            numbers_tuple = tuple(sorted(numbers))
            
            # 중복 체크 및 유효성 검사
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
    
    def _generate_by_frequency(self) -> List[int]:
        """빈도 편향 기반 생성"""
        # 가중치를 확률로 변환
        weights = [self.frequency_bias.get(n, 1.0) for n in range(1, self.max_number + 1)]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        
        numbers = []
        available = list(range(1, self.max_number + 1))
        
        for _ in range(self.ball_count):
            # 가중 랜덤 선택
            probs = [probabilities[n-1] if n in available else 0 for n in range(1, self.max_number + 1)]
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p / total_prob for p in probs]
            
            chosen = random.choices(range(1, self.max_number + 1), weights=probs, k=1)[0]
            numbers.append(chosen)
            available.remove(chosen)
        
        return numbers
    
    def _generate_by_position(self) -> List[int]:
        """위치 편향 기반 생성"""
        numbers = []
        available = list(range(1, self.max_number + 1))
        
        for pos in range(self.ball_count):
            pos_probs = self.position_bias.get(pos, {})
            weights = [pos_probs.get(n, 0.01) if n in available else 0 
                      for n in range(1, self.max_number + 1)]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            
            chosen = random.choices(range(1, self.max_number + 1), weights=weights, k=1)[0]
            numbers.append(chosen)
            if chosen in available:
                available.remove(chosen)
        
        return sorted(numbers)
    
    def _generate_hot_numbers(self) -> List[int]:
        """핫 번호 우선 생성"""
        # 핫 번호에서 일부, 나머지에서 일부
        hot_count = min(self.ball_count - 1, len(self.recent_hot))
        
        hot_picks = random.sample(self.recent_hot, hot_count)
        
        # 나머지 번호에서 선택
        remaining = [n for n in range(1, self.max_number + 1) if n not in hot_picks]
        other_picks = random.sample(remaining, self.ball_count - hot_count)
        
        return hot_picks + other_picks
    
    def _generate_balanced(self) -> List[int]:
        """혼합 전략"""
        strategy = random.choice(["frequency", "position", "hot"])
        
        if strategy == "frequency":
            return self._generate_by_frequency()
        elif strategy == "position":
            return self._generate_by_position()
        else:
            return self._generate_hot_numbers()
    
    def _analyze_numbers(self, numbers: List[int]) -> Dict:
        """생성된 번호 분석"""
        hot_count = sum(1 for n in numbers if n in self.recent_hot)
        cold_count = sum(1 for n in numbers if n in self.recent_cold)
        
        # 각 번호의 편향도 평균
        avg_bias = sum(self.frequency_bias.get(n, 1.0) for n in numbers) / len(numbers)
        
        return {
            "sum": sum(numbers),
            "hot_count": hot_count,
            "cold_count": cold_count,
            "bias_score": round(avg_bias, 2),
            "bias_rating": self._get_bias_rating(avg_bias)
        }
    
    def _get_bias_rating(self, bias: float) -> str:
        """편향도 등급"""
        if bias >= 1.2:
            return "높음 (편향 활용)"
        elif bias >= 1.0:
            return "보통"
        else:
            return "낮음 (비편향)"
    
    def get_hot_numbers(self, count: int = 10) -> List[int]:
        """핫 번호 반환"""
        return self.recent_hot[:count]
    
    def get_cold_numbers(self, count: int = 10) -> List[int]:
        """콜드 번호 반환"""
        return self.recent_cold[:count]


def create_model(draws: List[Dict], ball_count: int = 6, max_number: int = 45) -> PhysicsBiasModel:
    """모델 생성 헬퍼 함수"""
    return PhysicsBiasModel(draws, ball_count, max_number)
