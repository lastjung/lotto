"""
연속번호 분석
로또 번호에 연속된 숫자가 있는지 검사
"""


def has_consecutive(numbers: list, min_count: int = 3) -> bool:
    """
    연속 번호 감지 (기본: 3개 이상 연속이면 True)
    
    통계적 배경:
    - 3개 이상 연속 번호가 나오는 경우는 약 5% 미만
    - 4개 이상 연속은 매우 드묾 (1% 미만)
    - 연속번호 조합은 당첨 확률이 통계적으로 낮음
    
    Args:
        numbers: 6개 번호 리스트
        min_count: 연속 판정 기준 (기본 3개)
    
    Returns:
        True if min_count 이상 연속 번호 존재, False otherwise
    """
    sorted_nums = sorted(numbers)
    consecutive_count = 1
    
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] - sorted_nums[i-1] == 1:
            consecutive_count += 1
            if consecutive_count >= min_count:
                return True
        else:
            consecutive_count = 1
    
    return False


def get_consecutive_info(numbers: list) -> dict:
    """연속번호 정보 반환"""
    sorted_nums = sorted(numbers)
    sequences = []
    current_seq = [sorted_nums[0]]
    
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] - sorted_nums[i-1] == 1:
            current_seq.append(sorted_nums[i])
        else:
            if len(current_seq) >= 2:
                sequences.append(current_seq)
            current_seq = [sorted_nums[i]]
    
    if len(current_seq) >= 2:
        sequences.append(current_seq)
    
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    
    return {
        "has_consecutive_3": max_len >= 3,
        "max_consecutive": max_len,
        "sequences": sequences,
        "rating": _get_consecutive_rating(max_len)
    }


def _get_consecutive_rating(max_len: int) -> str:
    """연속번호 등급 반환"""
    if max_len <= 1:
        return "없음 (추천)"
    elif max_len == 2:
        return "2개 연속 (양호)"
    elif max_len == 3:
        return "3개 연속 (주의)"
    else:
        return f"{max_len}개 연속 (비추천)"
