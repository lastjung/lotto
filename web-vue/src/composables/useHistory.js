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

    /**
     * 히스토리 로드 (LocalStorage)
     */
    function loadHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY)
            if (raw) {
                const parsed = JSON.parse(raw)
                history.value = Array.isArray(parsed) ? parsed : []
            }
        } catch (e) {
            console.error('History load failed:', e)
            localStorage.removeItem(STORAGE_KEY)
            history.value = []
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
                numbers: entry.numbers,
                analysis: entry.analysis,
                generated_at: entry.date
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
        sortedHistory,
        loadHistory,
        saveEntry,
        clearHistory
    }
}
