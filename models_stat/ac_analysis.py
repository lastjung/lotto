"""
AC값(Arithmetic Complexity) 분석 모듈

AC값은 로또 번호 조합의 산술적 무작위성을 정량화한 지표입니다.
- 규칙적인 번호(1,2,3,4,5,6)는 낮은 AC값
- 무작위적인 번호는 높은 AC값
- 대부분의 당첨 번호는 AC값 7~10 구간에 분포

사용:
    from utils.ac_analysis import calculate_ac, is_valid_ac
    
    ac = calculate_ac([3, 15, 22, 29, 35, 42])
    valid = is_valid_ac([3, 15, 22, 29, 35, 42])  # AC >= 7 여부
"""

from itertools import combinations
from typing import List


def calculate_ac(numbers: List[int]) -> int:
    """
    AC값(Arithmetic Complexity) 계산
    
    알고리즘:
    1. 모든 두 수의 차이(Difference) 집합 D 계산
    2. D에서 중복 제거한 고유 원소 개수 U 계산
    3. AC = U - (N - 1)
    
    Args:
        numbers: 로또 번호 리스트 (예: [3, 15, 22, 29, 35, 42])
    
    Returns:
        AC값 (0 ~ C(N,2) - (N-1) 범위)
        
    Example:
        >>> calculate_ac([1, 2, 3, 4, 5, 6])  # 규칙적 → 낮은 AC
        0
        >>> calculate_ac([3, 15, 22, 29, 35, 42])  # 무작위 → 높은 AC
        10
    """
    n = len(numbers)
    if n < 2:
        return 0
    
    # 모든 두 수의 차이 계산
    differences = set()
    for a, b in combinations(sorted(numbers), 2):
        differences.add(abs(b - a))
    
    # 고유 차이 개수
    unique_count = len(differences)
    
    # AC = U - (N - 1)
    ac = unique_count - (n - 1)
    
    return ac


def is_valid_ac(numbers: List[int], min_ac: int = 7) -> bool:
    """
    AC값이 유효한지 검사 (규칙적인 조합 필터링)
    
    Args:
        numbers: 로또 번호 리스트
        min_ac: 최소 허용 AC값 (기본: 7)
    
    Returns:
        AC >= min_ac 이면 True
    """
    return calculate_ac(numbers) >= min_ac


def get_ac_rating(numbers: List[int]) -> str:
    """
    AC값에 따른 조합 품질 평가
    
    Args:
        numbers: 로또 번호 리스트
    
    Returns:
        품질 등급 문자열
    """
    ac = calculate_ac(numbers)
    
    if ac <= 3:
        return "매우 낮음 (피할 것)"
    elif ac <= 6:
        return "낮음 (비추천)"
    elif ac <= 8:
        return "보통 (괜찮음)"
    else:
        return "높음 (추천)"


def analyze_ac_distribution(numbers_list: List[List[int]]) -> dict:
    """
    여러 번호 조합의 AC값 분포 분석
    
    Args:
        numbers_list: 여러 회차의 번호 리스트
        
    Returns:
        AC값 통계 딕셔너리
    """
    if not numbers_list:
        return {}
    
    ac_values = [calculate_ac(nums) for nums in numbers_list]
    
    # AC값 분포
    distribution = {}
    for ac in ac_values:
        distribution[ac] = distribution.get(ac, 0) + 1
    
    return {
        "total_samples": len(ac_values),
        "min_ac": min(ac_values),
        "max_ac": max(ac_values),
        "avg_ac": sum(ac_values) / len(ac_values),
        "distribution": distribution,
        "low_ac_count": sum(1 for ac in ac_values if ac < 7),
        "high_ac_percent": sum(1 for ac in ac_values if ac >= 7) / len(ac_values) * 100
    }


# 테스트
if __name__ == "__main__":
    # 규칙적인 번호 (낮은 AC)
    regular = [1, 2, 3, 4, 5, 6]
    print(f"규칙적: {regular} → AC = {calculate_ac(regular)}, 유효: {is_valid_ac(regular)}")
    
    # 무작위 번호 (높은 AC)
    random_nums = [3, 15, 22, 29, 35, 42]
    print(f"무작위: {random_nums} → AC = {calculate_ac(random_nums)}, 유효: {is_valid_ac(random_nums)}")
    
    # 다양한 예시
    examples = [
        [5, 10, 15, 20, 25, 30],  # 5 간격
        [1, 7, 14, 21, 28, 35],   # 7 간격
        [2, 8, 19, 27, 38, 43],   # 무작위
    ]
    
    for nums in examples:
        ac = calculate_ac(nums)
        print(f"{nums} → AC = {ac}, 등급: {get_ac_rating(nums)}")
