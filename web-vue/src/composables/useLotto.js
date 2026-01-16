import { ref, computed } from 'vue'
import { supabase } from 'src/boot/supabase'

// Country flag mapping
const FLAGS = {
    KR: '🇰🇷',
    US: '🇺🇸',
    CA: '🇨🇦',
    JP: '🇯🇵'
}

// Default lottery options (will be replaced when config loads)
const DEFAULT_OPTIONS = [
    { id: 'korea_645', name: '🇰🇷 Korea Lotto 6/45', maxNum: 45, pickCount: 6 }
]

// Singleton state - shared across all components
const lotteryOptions = ref([...DEFAULT_OPTIONS])
const selectedLotteryId = ref(localStorage.getItem('s_lottery') || 'korea_645')
const draws = ref([])
const loading = ref(false)
const error = ref(null)
const configLoaded = ref(false)

const currentLottery = computed(() =>
    lotteryOptions.value.find(l => l.id === selectedLotteryId.value) || lotteryOptions.value[0]
)

/**
 * config/lotteries.json에서 로또 목록 로드
 */
async function loadLotteryConfig() {
    if (configLoaded.value) return
    
    try {
        const res = await fetch('/config/lotteries.json')
        if (!res.ok) throw new Error('Config load failed')
        
        const config = await res.json()
        
        // Convert config to options array
        lotteryOptions.value = Object.entries(config).map(([id, cfg]) => ({
            id,
            name: `${FLAGS[cfg.country] || '🌐'} ${cfg.name}`,
            maxNum: cfg.ball_range?.[1] || 45,
            pickCount: cfg.ball_count || 6,
            bonus: cfg.special_ball?.range?.[1] || (cfg.has_bonus ? cfg.ball_range?.[1] : null),
            country: cfg.country
        }))
        
        configLoaded.value = true
        console.log(`✅ Loaded ${lotteryOptions.value.length} lottery configs`)
    } catch (e) {
        console.warn('⚠️ Failed to load lottery config, using defaults:', e)
    }
}

/**
 * 로또 데이터 로드 (Supabase 우선, JSON 폴백)
 */
async function loadDraws(lotteryId = selectedLotteryId.value) {
    // Ensure config is loaded first
    await loadLotteryConfig()
    
    loading.value = true
    error.value = null

    try {
        // 1. Supabase 시도
        const { data, error: sbError } = await supabase
            .from('lotto_draws')
            .select('*')
            .eq('lottery_type', lotteryId)
            .order('draw_number', { ascending: false })

        if (sbError) throw sbError

        if (data && data.length > 0) {
            draws.value = data
            console.log(`✅ Loaded ${data.length} draws from Supabase (${lotteryId})`)
        } else {
            // 2. 데이터가 없으면 JSON 폴백 (임시)
            console.warn('⚠️ No data in Supabase, falling back to JSON')
            const response = await fetch(window.location.origin + `/data/${lotteryId}/draws.json`)
            if (!response.ok) throw new Error('Failed to load JSON data')
            const json = await response.json()
            draws.value = (json.draws || json).sort((a, b) => b.draw_number - a.draw_number)
            console.log(`✅ Loaded ${draws.value.length} draws from JSON (${lotteryId})`)
        }
    } catch (e) {
        console.error('❌ Data load failed:', e)
        error.value = e.message
    } finally {
        loading.value = false
    }
}

function selectLottery(id) {
    selectedLotteryId.value = id
    localStorage.setItem('s_lottery', id)
    loadDraws(id)
}

export function useLotto() {
    return {
        lotteryOptions,
        selectedLotteryId,
        currentLottery,
        draws,
        loading,
        error,
        loadDraws,
        selectLottery,
        loadLotteryConfig
    }
}
