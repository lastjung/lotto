/**
 * useApi - API 통신 composable
 * FastAPI 백엔드와의 모든 통신을 담당
 * 비로컬 환경에서는 Static Mode (ONNX 브라우저 추론) 사용
 */
import { ref, computed } from 'vue'

// 환경 감지 (web/js/app.js와 동일한 로직)
const IS_LOCALHOST = typeof window !== 'undefined' && 
    ['localhost', '127.0.0.1'].includes(window.location?.hostname)
const IS_DEV_PORT = typeof window !== 'undefined' && 
    ['8000', '9000'].includes(window.location?.port)

// Static Mode: 비로컬이거나 개발 포트가 아닌 경우 (Vercel 등 배포 환경)
const IS_STATIC_MODE = !IS_LOCALHOST || !IS_DEV_PORT
const API_BASE = IS_STATIC_MODE ? '' : 'http://localhost:8000'

// 상태 로깅
if (typeof window !== 'undefined') {
    console.log(`🌐 API Mode: ${IS_STATIC_MODE ? 'STATIC/ONNX' : 'API/SERVER'} (Base: ${API_BASE || 'local'})`)
}

export function useApi() {
    const loading = ref(false)
    const error = ref(null)

    /**
     * 번호 생성 API 호출
     * @param {string} lotteryId - 로또 종류 (korea_645, japan_loto6 등)
     * @param {string} modelType - 모델 종류 (transformer, lstm, vector, hot_trend)
     * @param {object} options - 옵션 { count, acFilter, sumFilter, consecutiveFilter }
     * @returns {Promise<object>} 생성 결과
     */
    async function generateNumbers(lotteryId, modelType, options = {}) {
        loading.value = true
        error.value = null

        try {
            const res = await fetch(`${API_BASE}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lottery_id: lotteryId,
                    count: options.count ?? 5,
                    model_type: modelType,
                    ac_filter: options.acFilter ?? false,
                    sum_filter: options.sumFilter ?? false,
                    consecutive_filter: options.consecutiveFilter ?? false
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
        loadLotteryData,
        isStaticMode: IS_STATIC_MODE,
        apiBase: API_BASE
    }
}
