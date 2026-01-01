<template>
  <div class="history-list">
    <!-- Empty State -->
    <div v-if="history.length === 0" class="empty-state text-center q-py-xl">
      <div class="text-4xl q-mb-sm">📜</div>
      <p class="text-grey-6">No generation history yet</p>
    </div>

    <!-- History Items -->
    <div
      v-for="entry in history"
      :key="entry.id"
      class="history-item q-mb-md"
    >
      <!-- Header -->
      <div class="item-header q-mb-sm">
        <div class="flex items-center gap-2">
          <q-badge color="green" size="sm" :label="entry.lottery_name || 'Korea Lotto'" />
          <q-badge color="purple" size="sm" :label="(entry.model || 'unknown').toUpperCase()" />
        </div>
        <span class="text-caption text-grey-6">
          {{ formatDate(entry.date || entry.generated_at) }}
        </span>
      </div>

      <!-- Number Sets -->
      <div class="number-sets">
        <div
          v-for="(nums, idx) in getNumbers(entry)"
          :key="idx"
          class="number-row q-mb-xs"
        >
          <span class="set-idx text-grey-7">#{{ idx + 1 }}</span>
          <div class="balls">
            <span
              v-for="n in nums"
              :key="n"
              class="ball-sm"
              :class="getBallColor(n)"
            >
              {{ n }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Clear Button -->
    <div v-if="history.length > 0" class="text-center q-mt-lg">
      <q-btn
        flat
        color="red-5"
        icon="delete_sweep"
        label="Clear History"
        @click="$emit('clear')"
      />
    </div>
  </div>
</template>

<script setup>
import { useLottery } from 'src/composables/useLottery'

defineProps({
  history: {
    type: Array,
    default: () => []
  }
})

defineEmits(['clear'])

const { getBallColor } = useLottery()

function formatDate(dateStr) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('ko-KR')
}

function getNumbers(entry) {
  if (!entry.numbers) return []
  return Array.isArray(entry.numbers) ? entry.numbers : []
}
</script>

<style scoped lang="scss">
.history-item {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-left: 3px solid rgba(168, 85, 247, 0.5);
  border-radius: 12px;
  padding: 16px;
  transition: background 0.2s;

  &:hover {
    background: rgba(255, 255, 255, 0.06);
  }
}

.item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}

.number-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.set-idx {
  font-size: 10px;
  min-width: 20px;
  font-family: monospace;
}

.balls {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.ball-sm {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: bold;
}

.bg-yellow-500 { background: #eab308; color: #000; }
.bg-blue-500 { background: #3b82f6; color: #fff; }
.bg-red-500 { background: #ef4444; color: #fff; }
.bg-gray-500 { background: #6b7280; color: #fff; }
.bg-green-500 { background: #22c55e; color: #fff; }
</style>
