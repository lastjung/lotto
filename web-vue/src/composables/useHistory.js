/**
 * useHistory - 히스토리 관리 composable
 * LocalStorage + Supabase 이중 저장
 */
import { ref, computed } from 'vue'
import { supabase } from 'src/boot/supabase'

const STORAGE_KEY = 'lotto_history'
const MAX_HISTORY = 100

export function useHistory() {
    const history = ref([])
    const loading = ref(false)

    /**
     * 히스토리 로드 (Supabase + LocalStorage)
     */
    async function loadHistory() {
        loading.value = true
        try {
            // 1. Supabase에서 최신 50개 로드
            const { data, error: sbError } = await supabase
                .from('lotto_history')
                .select('*')
                .order('created_at', { ascending: false })
                .limit(50)

            if (!sbError && data) {
                history.value = data.map(item => {
                    let rawNumbers = item.numbers
                    let finalNumbers = []
                    let finalAnalysis = null

                    // 1. 문자열일 경우 파싱 (여러 번 중첩된 경우까지 고려)
                    while (typeof rawNumbers === 'string') {
                        try {
                            rawNumbers = JSON.parse(rawNumbers)
                        } catch (e) {
                            console.warn('Failed to parse numbers string:', e)
                            break
                        }
                    }

                    // 2. 객체 구조 분석
                    if (rawNumbers && typeof rawNumbers === 'object') {
                        // case: { numbers: [...], analysis: {...} }
                        if (rawNumbers.numbers) {
                            finalNumbers = rawNumbers.numbers
                            finalAnalysis = rawNumbers.analysis || null
                        }
                        // case: [ { numbers: [...], analysis: {...} }, ... ] (배열 안의 객체)
                        else if (Array.isArray(rawNumbers) && rawNumbers[0] && typeof rawNumbers[0] === 'object') {
                            finalNumbers = rawNumbers[0].numbers || []
                            finalAnalysis = rawNumbers[0].analysis || null
                        }
                        // case: [1, 2, 3...] (순수 배열)
                        else if (Array.isArray(rawNumbers)) {
                            finalNumbers = rawNumbers
                        }
                        // case: { ... } (기타 객체)
                        else {
                            finalAnalysis = rawNumbers
                        }
                    }

                    return {
                        id: item.id,
                        date: item.draw_date || item.created_at,
                        model: item.model,
                        lottery_type: item.lottery_type,
                        numbers: Array.isArray(finalNumbers) ? finalNumbers : [],
                        analysis: finalAnalysis
                    }
                })
                console.log('✅ History loaded and normalized from Supabase')
                return
            }

            // 2. 실패시 또는 데이터 없을시 LocalStorage 폴백
            const raw = localStorage.getItem(STORAGE_KEY)
            if (raw) {
                const parsed = JSON.parse(raw)
                history.value = Array.isArray(parsed) ? parsed : []
            }
        } catch (e) {
            console.error('History load failed:', e)
            history.value = []
        } finally {
            loading.value = false
        }
    }

    /**
     * 엔트리 저장
     * @param {object} data - { numbers, model, lotteryType, lotteryName }
     */
    async function saveEntry(data) {
        const entry = {
            id: Date.now(),
            date: new Date().toISOString(),
            model: data.model,
            lottery_type: data.lotteryType,
            lottery_name: data.lotteryName,
            numbers: data.numbers,
            analysis: data.analysis || null
        }

        // 1. LocalStorage
        try {
            const current = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
            current.unshift(entry)
            localStorage.setItem(STORAGE_KEY, JSON.stringify(current.slice(0, MAX_HISTORY)))
            history.value = current.slice(0, MAX_HISTORY)
            console.log('✅ LocalStorage saved')
        } catch (e) {
            console.error('LocalStorage save failed:', e)
        }

        // 2. Supabase
        try {
            const { error } = await supabase.from('lotto_history').insert([{
                lottery_type: entry.lottery_type,
                model: entry.model,
                numbers: { numbers: entry.numbers, analysis: entry.analysis },
                user_id: localStorage.getItem('lotto_user_id') || 'guest',
                draw_date: entry.date
            }])

            if (error) {
                console.warn('Supabase save error:', error.message)
            } else {
                console.log('✅ Supabase saved')
            }
        } catch (e) {
            console.warn('Supabase save failed:', e)
        }
    }

    /**
     * 히스토리 초기화
     */
    function clearHistory() {
        localStorage.removeItem(STORAGE_KEY)
        history.value = []
    }

    // 최신순 정렬된 히스토리
    const sortedHistory = computed(() => {
        return [...history.value].sort((a, b) => new Date(b.date) - new Date(a.date))
    })

    return {
        history,
        loading,
        sortedHistory,
        loadHistory,
        saveEntry,
        clearHistory
    }
}
