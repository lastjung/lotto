/**
 * useLottery - 로또 선택 및 데이터 관리 composable
 */
import { ref, computed } from 'vue'

// 지원하는 로또 종류
const LOTTERY_OPTIONS = [
    { id: 'korea_645', name: 'Korea Lotto 6/45', maxNum: 45, pickCount: 6 },
    { id: 'japan_loto6', name: 'Japan Loto 6', maxNum: 43, pickCount: 6 },
    { id: 'canada_649', name: 'Canada Lotto 6/49', maxNum: 49, pickCount: 6 },
    { id: 'us_powerball', name: 'US Powerball', maxNum: 69, pickCount: 5, bonus: 26 },
    { id: 'us_megamillions', name: 'US Mega Millions', maxNum: 70, pickCount: 5, bonus: 25 }
]

// 지원하는 모델 종류
const MODEL_OPTIONS = [
    { id: 'transformer', name: 'Transformer', icon: '⚡', description: 'Attention-based Pattern Recognition' },
    { id: 'lstm', name: 'LSTM', icon: '🔮', description: 'Sequential Time-Series Analysis' },
    { id: 'vector', name: 'Physics', icon: '🎱', description: 'Vector-based Bias Detection' },
    { id: 'hot_trend', name: 'Hot Trend', icon: '🔥', description: 'Frequency-based Weighted Random' }
]

export function useLottery() {
    // 상태
    const selectedLottery = ref(localStorage.getItem('s_lottery') || 'korea_645')
    const selectedModel = ref(localStorage.getItem('s_model') || 'transformer')
    const lotteryData = ref([])

    // 현재 선택된 로또 정보
    const currentLottery = computed(() => {
        return LOTTERY_OPTIONS.find(l => l.id === selectedLottery.value) || LOTTERY_OPTIONS[0]
    })

    // 현재 선택된 모델 정보
    const currentModel = computed(() => {
        return MODEL_OPTIONS.find(m => m.id === selectedModel.value) || MODEL_OPTIONS[0]
    })

    /**
     * 로또 선택 변경
     * @param {string} lotteryId
     */
    function selectLottery(lotteryId) {
        selectedLottery.value = lotteryId
        localStorage.setItem('s_lottery', lotteryId)
    }

    /**
     * 모델 선택 변경
     * @param {string} modelId
     */
    function selectModel(modelId) {
        selectedModel.value = modelId
        localStorage.setItem('s_model', modelId)
    }

    /**
     * 최근 N회차 데이터 가져오기
     * @param {number} count
     * @returns {Array}
     */
    function getRecentDraws(count = 30) {
        if (!lotteryData.value || lotteryData.value.length === 0) return []
        return lotteryData.value.slice(-count).reverse()
    }

    /**
     * 공 색상 클래스 반환 (한국 로또 기준)
     * @param {number} num
     * @returns {string}
     */
    function getBallColor(num) {
        if (num <= 10) return 'bg-yellow-500 text-black'
        if (num <= 20) return 'bg-blue-500 text-white'
        if (num <= 30) return 'bg-red-500 text-white'
        if (num <= 40) return 'bg-gray-500 text-white'
        return 'bg-green-500 text-white'
    }

    return {
        // 옵션
        lotteryOptions: LOTTERY_OPTIONS,
        modelOptions: MODEL_OPTIONS,

        // 상태
        selectedLottery,
        selectedModel,
        lotteryData,

        // Computed
        currentLottery,
        currentModel,

        // 액션
        selectLottery,
        selectModel,
        getRecentDraws,
        getBallColor
    }
}
