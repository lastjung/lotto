"""
Collectors package for lottery data collection.
"""

from .base import BaseLotteryCollector
from .korea_645 import Korea645Collector
from .usa_powerball import USAPowerballCollector
from .usa_megamillions import USAMegaMillionsCollector
from .canada_649 import CanadaLottoCollector
from .japan_loto6 import JapanLoto6Collector

# 수집기 레지스트리
COLLECTORS = {
    "korea_645": Korea645Collector,
    "usa_powerball": USAPowerballCollector,
    "usa_megamillions": USAMegaMillionsCollector,
    "canada_649": CanadaLottoCollector,
    "japan_loto6": JapanLoto6Collector,
}


def get_collector(lottery_id: str, config: dict) -> BaseLotteryCollector:
    """로또 ID로 적절한 수집기를 반환합니다."""
    collector_class = COLLECTORS.get(lottery_id)
    if collector_class is None:
        raise ValueError(f"Unknown lottery: {lottery_id}")
    return collector_class(lottery_id, config)


def list_available_collectors() -> list[str]:
    """사용 가능한 수집기 목록을 반환합니다."""
    return list(COLLECTORS.keys())
