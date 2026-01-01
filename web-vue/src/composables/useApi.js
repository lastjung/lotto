/**
 * useApi - API 통신 composable
 * FastAPI 백엔드와의 모든 통신을 담당
 */
import { ref } from 'vue'

const API_BASE = 'http://localhost:8000'

export function useApi() {
    const loading = ref(false)
    const error = ref(null)

    /**
     * 번호 생성 API 호출
     * @param {string} lotteryId - 로또 종류 (korea_645, japan_loto6 등)
     * @param {string} modelType - 모델 종류 (transformer, lstm, vector, hot_trend)
     * @param {object} filters - 필터 옵션 { acFilter, sumFilter, consecutiveFilter }
     * @returns {Promise<object>} 생성 결과
     */
    async function generateNumbers(lotteryId, modelType, filters = {}) {
        loading.value = true
        error.value = null

        try {
            const res = await fetch(`${API_BASE}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lottery_id: lotteryId,
                    count: 5,
                    model_type: modelType,
                    ac_filter: filters.acFilter ?? false,
                    sum_filter: filters.sumFilter ?? false,
                    consecutive_filter: filters.consecutiveFilter ?? false
                })
            })

            if (!res.ok) {
                throw new Error(`API Error: ${res.status}`)
            }

            return await res.json()
        } catch (e) {
            error.value = e.message
            throw e
        } finally {
            loading.value = false
        }
    }

    /**
     * 설정 저장 API
     * @param {string} lottery - 로또 종류
     * @param {string} model - 모델 종류
     */
    async function saveConfig(lottery, model) {
        try {
            await fetch(`${API_BASE}/api/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    default_lottery: lottery,
                    default_model: model,
                    updated_at: new Date().toISOString()
                })
            })
        } catch (e) {
            console.warn('Config save failed:', e)
        }
    }

    /**
     * 로또 데이터 로드
     * @param {string} lotteryId - 로또 종류
     * @returns {Promise<Array>} 추첨 데이터 배열
     */
    async function loadLotteryData(lotteryId) {
        try {
            const res = await fetch(`${API_BASE}/api/data/${lotteryId}`)
            if (!res.ok) throw new Error('Data load failed')
            return await res.json()
        } catch (e) {
            console.error('Failed to load lottery data:', e)
            return []
        }
    }

    return {
        loading,
        error,
        generateNumbers,
        saveConfig,
        loadLotteryData
    }
}
