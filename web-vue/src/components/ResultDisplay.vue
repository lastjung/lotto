<template>
  <div class="result-display">
    <!-- Header -->
    <div class="result-header q-mb-md">
      <div class="flex items-center gap-2">
        <q-badge color="green" :label="lotteryName" />
        <q-badge color="purple" :label="modelName.toUpperCase()" />
        <span class="text-grey-6 text-caption">Draw #{{ drawNumber }}</span>
      </div>
      <div class="text-caption text-grey-6">{{ formattedDate }}</div>
    </div>

    <!-- Number Sets -->
    <div class="number-sets q-mb-md">
      <div
        v-for="(set, idx) in numbers"
        :key="idx"
        class="number-row q-mb-sm"
      >
        <span class="set-number text-grey-6">#{{ idx + 1 }}</span>
        <div class="balls">
          <span
            v-for="num in set"
            :key="num"
            class="ball"
            :class="getBallColor(num)"
          >
            {{ num }}
          </span>
        </div>
      </div>
    </div>

    <!-- Analysis (Optional) -->
    <div v-if="analysis" class="analysis-section">
      <q-separator class="q-my-md" dark />
      <div class="text-caption text-grey-5 q-mb-sm">📊 Analysis</div>
      <div class="flex gap-4 flex-wrap">
        <q-chip size="sm" color="blue-grey-8" text-color="white" icon="functions">
          Sum: {{ analysis.sum }}
        </q-chip>
        <q-chip size="sm" color="blue-grey-8" text-color="white" icon="analytics">
          AC: {{ analysis.ac_value }} ({{ analysis.ac_rating }})
        </q-chip>
        <q-chip v-if="analysis.odd_even" size="sm" color="blue-grey-8" text-color="white">
          Odd/Even: {{ analysis.odd_even }}
        </q-chip>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useLottery } from 'src/composables/useLottery'

const props = defineProps({
  numbers: {
    type: Array,
    default: () => []
  },
  modelName: {
    type: String,
    default: 'Unknown'
  },
  lotteryName: {
    type: String,
    default: 'Korea Lotto'
  },
  drawNumber: {
    type: [String, Number],
    default: '-'
  },
  analysis: {
    type: Object,
    default: null
  },
  generatedAt: {
    type: String,
    default: null
  }
})

const { getBallColor } = useLottery()

const formattedDate = computed(() => {
  if (!props.generatedAt) return new Date().toLocaleString('ko-KR')
  return new Date(props.generatedAt).toLocaleString('ko-KR')
})
</script>

<style scoped lang="scss">
.result-display {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 20px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.number-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.set-number {
  min-width: 24px;
  font-size: 10px;
  font-family: monospace;
}

.balls {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.ball {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.bg-yellow-500 { background: #eab308; color: #000; }
.bg-blue-500 { background: #3b82f6; color: #fff; }
.bg-red-500 { background: #ef4444; color: #fff; }
.bg-gray-500 { background: #6b7280; color: #fff; }
.bg-green-500 { background: #22c55e; color: #fff; }
</style>
