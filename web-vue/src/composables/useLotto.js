import { ref, computed } from 'vue'
import { supabase } from 'src/boot/supabase'

// 지원하는 로또 종류
const LOTTERY_OPTIONS = [
    { id: 'korea_645', name: 'Korea Lotto 6/45', maxNum: 45, pickCount: 6, icon: 'flag' },
    { id: 'usa_powerball', name: 'USA Powerball', maxNum: 69, pickCount: 5, bonus: 26, icon: 'public' },
    { id: 'usa_megamillions', name: 'USA Mega Millions', maxNum: 70, pickCount: 5, bonus: 25, icon: 'public' },
    { id: 'canada_649', name: 'Canada Lotto 6/49', maxNum: 49, pickCount: 6, icon: 'public' }
]

export function useLotto() {
    const selectedLotteryId = ref(localStorage.getItem('s_lottery') || 'korea_645')
    const draws = ref([])
    const loading = ref(false)
    const error = ref(null)

    const currentLottery = computed(() =>
        LOTTERY_OPTIONS.find(l => l.id === selectedLotteryId.value) || LOTTERY_OPTIONS[0]
    )

    /**
     * 로또 데이터 로드 (Supabase 우선, JSON 폴백)
     */
    async function loadDraws(lotteryId = selectedLotteryId.value) {
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

    return {
        lotteryOptions: LOTTERY_OPTIONS,
        selectedLotteryId,
        currentLottery,
        draws,
        loading,
        error,
        loadDraws,
        selectLottery
    }
}
