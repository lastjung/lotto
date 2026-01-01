/**
 * useHistory - 히스토리 관리 composable
 * LocalStorage + Supabase 이중 저장
 */
import { ref, computed } from 'vue'
import { createClient } from '@supabase/supabase-js'

// Supabase 설정
const SB_URL = 'https://sfqlshdlqwqlkxdrfdke.supabase.co'
const SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNmcWxzaGRscXdxbGt4ZHJmZGtlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5MDM0NzUsImV4cCI6MjA4MTQ3OTQ3NX0.CMbJ_5IUxAifoNIzqdxu_3sz31AtOMw2vRBPxfxZzSk'

let supabaseClient = null

function getSupabase() {
    if (!supabaseClient) {
        supabaseClient = createClient(SB_URL, SB_KEY)
    }
    return supabaseClient
}

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
            const supabase = getSupabase()
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
