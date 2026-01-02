<template>
  <div class="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-8 pb-24 md:pb-8 scroll-smooth" id="main-scroll">
    <div id="view-dashboard" class="space-y-6 max-w-[1600px] mx-auto">

      <!-- TOP ROW: VISUALIZATIONS -->
      <LottoCharts :draws="draws" :max-num="currentLottery.maxNum" />

      <!-- BOTTOM ROW: CONFIG (Left) & RESULTS (Right) -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- LEFT COL: CONFIGURATION (Span 4) -->
        <div class="lg:col-span-5 xl:col-span-4 space-y-6">

          <!-- 1. AI Analysis Engine (Unified 2x2 Grid) -->
          <div>
            <h3 class="text-lg font-bold text-white flex items-center gap-2 mb-4">
              <span class="text-pink-400 text-2xl">🧠</span>
              AI Analysis Engine
            </h3>

            <div class="grid grid-cols-2 gap-3">
              <div 
                v-for="model in aiModels" 
                :key="model.id"
                @click="selectModel(model.id)"
                class="model-card p-4 rounded-xl bg-[#1e293b] border transition-all text-left flex flex-col justify-between h-36 group relative overflow-hidden cursor-pointer"
                :class="selectedModel === model.id ? 'border-blue-500 bg-blue-500/10' : 'border-blue-500/30 hover:border-blue-500 hover:bg-blue-500/10'"
              >
                <div class="w-8 h-8 rounded bg-blue-500/20 text-blue-400 flex items-center justify-center text-lg mb-2 group-hover:scale-110 transition-transform">
                  {{ model.icon }}
                </div>
                <div>
                  <div class="font-bold text-white text-sm">{{ model.name }}</div>
                  <div class="text-[10px] text-gray-500 leading-tight mt-1">{{ model.desc }}</div>
                </div>
                <!-- Active Dot -->
                <div v-if="selectedModel === model.id" class="absolute top-3 right-3 w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_10px_rgba(6,182,212,0.5)]"></div>
              </div>
            </div>
          </div>

          <!-- 2. Statistical Models -->
          <div>
            <h3 class="text-lg font-bold text-white flex items-center gap-2 mb-4">
              <span class="text-blue-400 text-2xl">📊</span>
              Statistical Models
            </h3>

            <div class="grid grid-cols-2 gap-3">
              <div 
                v-for="model in statModels" 
                :key="model.id"
                @click="selectModel(model.id)"
                class="model-card p-4 rounded-xl bg-[#1e293b] border transition-all text-left flex flex-col justify-between h-32 group relative overflow-hidden cursor-pointer"
                :class="selectedModel === model.id ? 'border-purple-500 bg-purple-500/10' : 'border-white/10 hover:border-purple-500 hover:bg-purple-500/10'"
              >
                <div class="w-8 h-8 rounded bg-purple-500/20 text-purple-400 flex items-center justify-center text-lg mb-2 group-hover:scale-110 transition-transform">
                  {{ model.icon }}
                </div>
                <div>
                  <div class="font-bold text-white text-sm">{{ model.name }}</div>
                  <div class="text-[10px] text-gray-500 leading-tight mt-1">{{ model.desc }}</div>
                </div>
                <div v-if="selectedModel === model.id" class="absolute top-3 right-3 w-2 h-2 bg-purple-500 rounded-full shadow-[0_0_10px_rgba(168,85,247,0.5)]"></div>
              </div>
            </div>
          </div>

        </div>

        <!-- RIGHT COL: RESULTS AREA (Span 8) -->
        <div class="lg:col-span-7 xl:col-span-8">
          <LottoResultArea
            :results="generatedNumbers"
            :analysis="currentAnalysis"
            :generating="loading"
            :scanning="scanning"
            :model-name="selectedModelName"
            @generate="generate"
          />
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useLotto } from 'src/composables/useLotto'
import { useHistory } from 'src/composables/useHistory'
import { useAiEngine } from 'src/composables/useAiEngine'
import LottoResultArea from 'src/components/LottoResultArea.vue'
import LottoCharts from 'src/components/LottoCharts.vue'

const { currentLottery, draws, loading: dataLoading, loadDraws } = useLotto()
const { saveEntry } = useHistory()
const { 
  loadModel, 
  generateWithAi, 
  generateWithStat, 
  loading: aiLoading, 
  modelLoaded 
} = useAiEngine()

const loading = ref(false)
const scanning = ref(false)
const selectedModel = ref('transformer')
const generatedNumbers = ref(null)
const currentAnalysis = ref(null)

const aiModels = [
  { id: 'transformer', name: 'Transformer', icon: '⚖️', desc: 'Attention-based Pattern Recognition' },
  { id: 'lstm', name: 'LSTM', icon: '⚡', desc: 'Sequential Time-Series Analysis' },
  { id: 'vector', name: 'AI Vector', icon: '🌌', desc: 'Vector-based Bias Detection' },
  { id: 'hybrid', name: 'Hybrid Pro', icon: '🧬', desc: 'Combined Neural & Statistical' }
]

const statModels = [
  { id: 'balanced_mix', name: 'Balanced Mix', icon: '⚖️', desc: 'Std. Deviation Focus' },
  { id: 'hot_trend', name: 'Hot Trend', icon: '🔥', desc: 'Recent Frequency' },
  { id: 'cold_theory', name: 'Cold Theory', icon: '🧊', desc: 'Target Overdue' },
  { id: 'physics_bias', name: 'Physics Bias', icon: '🎱', desc: 'Mechanical Imperfections' }
]

const selectedModelName = computed(() => {
  const all = [...aiModels, ...statModels]
  return all.find(m => m.id === selectedModel.value)?.name || 'Unknown'
})

onMounted(async () => {
  if (draws.value.length === 0) {
    await loadDraws()
  }
  
  setTimeout(async () => {
      try {
        await loadModel(currentLottery.value.id, selectedModel.value)
      } catch (e) {
        console.warn('⚠️ Non-fatal AI Init Error:', e)
      }
  }, 100)
})

watch([selectedModel, () => currentLottery.value.id], async ([newModel, newLotteryId]) => {
  await loadModel(newLotteryId, newModel)
})

function selectModel (id) {
  selectedModel.value = id
}

async function generate () {
  if (!modelLoaded.value && !['transformer', 'lstm'].includes(selectedModel.value)) {
    await loadModel(currentLottery.value.id, selectedModel.value)
  }

  loading.value = true
  scanning.value = true
  generatedNumbers.value = null
  
  try {
    let results = []
    if (['transformer', 'lstm'].includes(selectedModel.value)) {
      results = await generateWithAi(currentLottery.value, draws.value)
    } else {
      results = await generateWithStat(selectedModel.value, currentLottery.value, draws.value)
    }

    setTimeout(async () => {
      loading.value = false
      scanning.value = false
      
      const topSet = results[0]
      generatedNumbers.value = topSet
      
      currentAnalysis.value = {
        sum: topSet.reduce((a, b) => a + b, 0),
        ac_value: calculateAC(topSet),
        odd_even: calculateOddEven(topSet)
      }

      await saveEntry({
        numbers: generatedNumbers.value,
        model: selectedModel.value,
        lotteryType: currentLottery.value.id,
        lotteryName: currentLottery.value.name,
        analysis: currentAnalysis.value
      })
    }, 2000)

  } catch (e) {
    console.error('Generation failed:', e)
    loading.value = false
    scanning.value = false
  }
}

function calculateAC(numbers) {
  const sorted = [...numbers].sort((a, b) => a - b)
  const diffs = new Set()
  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) {
      diffs.add(sorted[j] - sorted[i])
    }
  }
  return diffs.size - (numbers.length - 1)
}

function calculateOddEven(numbers) {
  const odd = numbers.filter(n => n % 2 !== 0).length
  const even = numbers.length - odd
  return `${odd}:${even}`
}
</script>

<style lang="scss" scoped>
.model-card {
  /* Legacy hover effect styles are partly handled by Tailwind classes in template, 
     but keeping the transition here for smoothness if needed. */
  transition: all 0.3s ease;
}
</style>
