"""
합계 필터 분석
로또 번호의 합계가 통계적으로 유효한 범위 내에 있는지 검사
"""


def is_valid_sum(numbers: list, min_sum: int = 100, max_sum: int = 175) -> bool:
    """
    합계 범위 검증 (기본: 100~175)
    
    통계적 배경:
    - 로또 6/45 이론적 합계 범위: 21 (1+2+3+4+5+6) ~ 255 (40+41+42+43+44+45)
    - 실제 당첨번호 평균: 약 130~140
    - 권장 범위: 100~175 (전체 당첨의 약 85% 커버)
    
    Args:
        numbers: 번호 리스트
        min_sum: 최소 합계 (기본 100)
        max_sum: 최대 합계 (기본 175)
    
    Returns:
        True if 합계가 범위 내, False otherwise
    """
    total = sum(numbers)
    return min_sum <= total <= max_sum


def is_valid_sum_dynamic(numbers: list, ball_count: int = 6, max_number: int = 45) -> bool:
    """
    동적 합계 범위 검증 (로또별로 다른 범위 적용)
    
    이론적 합계 범위 계산:
    - 최소: 1 + 2 + ... + ball_count
    - 최대: (max_number - ball_count + 1) + ... + max_number
    - 평균: (최소 + 최대) / 2
    - 권장 범위: 평균 ± 25%
    
    Args:
        numbers: 번호 리스트
        ball_count: 공 개수 (5 or 6)
        max_number: 최대 번호 (45, 49, 69 등)
    
    Returns:
        True if 합계가 범위 내
    """
    total = sum(numbers)
    
    # 이론적 합계 범위 계산
    min_theoretical = sum(range(1, ball_count + 1))  # 1+2+...+ball_count
    max_theoretical = sum(range(max_number - ball_count + 1, max_number + 1))
    avg = (min_theoretical + max_theoretical) / 2
    
    # 권장 범위: 평균 ± 30%
    margin = avg * 0.30
    min_sum = int(avg - margin)
    max_sum = int(avg + margin)
    
    return min_sum <= total <= max_sum


def get_sum_stats(numbers: list) -> dict:
    """합계 통계 반환"""
    total = sum(numbers)
    return {
        "sum": total,
        "in_range": is_valid_sum(numbers),
        "rating": _get_sum_rating(total)
    }


def _get_sum_rating(total: int) -> str:
    """합계 등급 반환"""
    if 115 <= total <= 165:
        return "최적 (추천)"
    elif 100 <= total <= 175:
        return "양호"
    elif 80 <= total <= 195:
        return "보통"
    else:
        return "극단값 (비추천)"
