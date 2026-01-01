<template>
  <q-page class="q-pa-md">
    <!-- Header -->
    <div class="text-center q-mb-lg">
      <h1 class="text-h4 text-weight-bold q-mb-sm">
        🎱 AI Lotto Analyzer
      </h1>
      <p class="text-grey-6">Deep Learning Powered Number Generation</p>
    </div>

    <!-- Lottery Selector -->
    <div class="row justify-center q-mb-lg">
      <LotterySelector
        v-model="selectedLottery"
        :options="lotteryOptions"
        @change="handleLotteryChange"
      />
    </div>

    <!-- Model Cards -->
    <div class="row q-col-gutter-md justify-center q-mb-xl">
      <div v-for="model in modelOptions" :key="model.id" class="col-6 col-md-3">
        <ModelCard
          :model="model"
          :selected-model-id="selectedModel"
          @select="handleModelSelect"
        />
      </div>
    </div>

    <!-- Generate Button -->
    <div class="text-center q-mb-xl">
      <q-btn
        color="purple"
        size="lg"
        icon="auto_awesome"
        label="Generate Numbers"
        :loading="loading"
        @click="generate"
      />
    </div>

    <!-- Results -->
    <div v-if="lastResult" class="q-mb-xl">
      <ResultDisplay
        :numbers="lastResult.numbers"
        :model-name="currentModel.name"
        :lottery-name="currentLottery.name"
        :draw-number="lastResult.draw_number || '-'"
        :analysis="lastResult.analysis"
        :generated-at="lastResult.generated_at"
      />
    </div>

    <!-- Error Message -->
    <div v-if="error" class="text-center q-mb-md">
      <q-banner rounded class="bg-red-9 text-white">
        <template #avatar>
          <q-icon name="error" color="white" />
        </template>
        {{ error }}
      </q-banner>
    </div>

    <!-- History Section -->
    <q-separator class="q-my-xl" dark />

    <div class="text-h6 q-mb-md flex items-center gap-2">
      <q-icon name="history" />
      Generation History
    </div>

    <HistoryList
      :history="sortedHistory"
      @clear="handleClearHistory"
    />
  </q-page>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useApi } from 'src/composables/useApi'
import { useHistory } from 'src/composables/useHistory'
import { useLottery } from 'src/composables/useLottery'

import ModelCard from 'src/components/ModelCard.vue'
import ResultDisplay from 'src/components/ResultDisplay.vue'
import HistoryList from 'src/components/HistoryList.vue'
import LotterySelector from 'src/components/LotterySelector.vue'

// Composables
const { generateNumbers, saveConfig, loading, error } = useApi()
const { loadHistory, saveEntry, clearHistory, sortedHistory } = useHistory()
const {
  lotteryOptions,
  modelOptions,
  selectedLottery,
  selectedModel,
  currentLottery,
  currentModel,
  selectLottery,
  selectModel
} = useLottery()

// State
const lastResult = ref(null)

// Lifecycle
onMounted(() => {
  loadHistory()
})

// Handlers
function handleLotteryChange(lotteryId) {
  selectLottery(lotteryId)
  saveConfig(lotteryId, selectedModel.value)
}

function handleModelSelect(modelId) {
  selectModel(modelId)
  saveConfig(selectedLottery.value, modelId)
}

async function generate() {
  try {
    const result = await generateNumbers(
      selectedLottery.value,
      selectedModel.value,
      {
        acFilter: false,
        sumFilter: false,
        consecutiveFilter: false
      }
    )

    lastResult.value = result

    // Save to history
    await saveEntry({
      numbers: result.numbers,
      model: selectedModel.value,
      lotteryType: selectedLottery.value,
      lotteryName: currentLottery.value.name,
      analysis: result.analysis
    })
  } catch (e) {
    console.error('Generation failed:', e)
  }
}

function handleClearHistory() {
  if (confirm('Delete all history?')) {
    clearHistory()
  }
}
</script>

<style scoped lang="scss">
.q-page {
  max-width: 900px;
  margin: 0 auto;
}
</style>
