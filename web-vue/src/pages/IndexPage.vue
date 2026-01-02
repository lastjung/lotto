<template>
  <q-page class="q-pa-lg">
    <div class="max-w-[1600px] mx-auto space-y-6">
      <!-- TOP ROW: VISUALIZATIONS -->
      <LottoCharts :draws="draws" />

      <!-- BOTTOM ROW: CONFIG & RESULTS -->
      <div class="row q-col-gutter-lg">
        <!-- LEFT COL: CONFIGURATION -->
        <div class="col-12 col-lg-5 col-xl-4 space-y-6">
          <!-- AI Analysis Engine -->
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
                class="model-card p-4 rounded-xl cursor-pointer transition-all h-36 border relative overflow-hidden"
                :class="selectedModel === model.id ? 'border-blue-500 bg-blue-500/10' : 'bg-[#1e293b] border-blue-500/30 hover:border-blue-500'"
              >
                <div class="w-8 h-8 rounded bg-blue-500/20 text-blue-400 flex items-center justify-center text-lg mb-2">
                  {{ model.icon }}
                </div>
                <div>
                  <div class="font-bold text-white text-sm">{{ model.name }}</div>
                  <div class="text-[10px] text-gray-500 leading-tight mt-1">{{ model.desc }}</div>
                </div>
                <div v-if="selectedModel === model.id" class="absolute top-3 right-3 w-2 h-2 bg-blue-500 rounded-full shadow-glow-cyan"></div>
              </div>
            </div>
          </div>

          <!-- Statistical Models -->
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
                class="model-card p-4 rounded-xl cursor-pointer transition-all h-32 border relative overflow-hidden"
                :class="selectedModel === model.id ? 'border-purple-500 bg-purple-500/10' : 'bg-[#1e293b] border-white/10 hover:border-purple-500'"
              >
                <div class="w-8 h-8 rounded bg-purple-500/20 text-purple-400 flex items-center justify-center text-lg mb-2">
                  {{ model.icon }}
                </div>
                <div>
                  <div class="font-bold text-white text-sm">{{ model.name }}</div>
                  <div class="text-[10px] text-gray-500 leading-tight mt-1">{{ model.desc }}</div>
                </div>
                <div v-if="selectedModel === model.id" class="absolute top-3 right-3 w-2 h-2 bg-purple-500 rounded-full shadow-glow-purple"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- RIGHT COL: RESULTS AREA -->
        <div class="col-12 col-lg-7 col-xl-8">
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
  </q-page>
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
  { id: 'vector', name: 'AI Vector', icon: '🌌', desc: 'Dimensional Relation Analysis' }
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
  
  // Await next tick to ensure UI renders first
  setTimeout(async () => {
      try {
        console.log('🤖 Initializing AI Model...')
        await loadModel(currentLottery.value.id, selectedModel.value)
      } catch (e) {
        console.warn('⚠️ Non-fatal AI Init Error:', e)
      }
  }, 100)
})

// Update model when selection or lottery changes
watch([selectedModel, () => currentLottery.value.id], async ([newModel, newLotteryId]) => {
  await loadModel(newLotteryId, newModel)
})

function selectModel (id) {
  selectedModel.value = id
}

async function generate () {
  if (!modelLoaded.value && !['transformer', 'lstm'].includes(selectedModel.value)) {
    // If not loaded and it's a model that needs loading
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

    // Wrap in dummy timeout for scanning effect
    setTimeout(async () => {
      loading.value = false
      scanning.value = false
      
      // Select one set (or top result)
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

// Helpers
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
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  &:hover {
    transform: translateY(-2px);
  }
}

.shadow-glow-cyan {
  box-shadow: 0 0 10px #3b82f6;
}

.shadow-glow-purple {
  box-shadow: 0 0 10px #a855f7;
}

.shadow-glow-pink {
  box-shadow: 0 0 20px rgba(236, 72, 153, 0.6);
}

.btn-glow-primary {
  background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
  box-shadow: 0 0 12px rgba(168, 85, 247, 0.4);
  &:hover {
    box-shadow: 0 0 20px rgba(236, 72, 153, 0.6);
    transform: translateY(-1px);
  }
}

.grid {
  display: grid;
}
</style>
