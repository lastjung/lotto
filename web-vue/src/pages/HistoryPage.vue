<template>
  <div class="max-w-7xl mx-auto p-4 md:p-8">
    <!-- Tab Navigation (Legacy Style) -->
    <div class="flex gap-2 mb-6 border-b border-white/10 pb-2">
      <button 
        @click="activeTab = 'draws'" 
        class="px-4 py-2 rounded-t-lg text-sm font-bold transition-all border-b-2"
        :class="activeTab === 'draws' ? 'bg-blue-600/20 text-blue-400 border-blue-500' : 'text-gray-400 hover:text-white hover:bg-white/5 border-transparent'"
      >
        📊 History Statistics
      </button>
      <button 
        @click="activeTab = 'history'" 
        class="px-4 py-2 rounded-t-lg text-sm font-medium transition-all"
        :class="activeTab === 'history' ? 'bg-white/10 text-white' : 'text-gray-400 hover:text-white hover:bg-white/5'"
      >
        📂 Generated History
      </button>
    </div>

    <!-- Tab 1: Draws Analysis -->
    <div v-if="activeTab === 'draws'" class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold mb-1 text-white">📊 Statistics</h1>
        <div class="flex flex-col sm:flex-row gap-2 text-sm text-gray-400">
          <p>Based on recent 1 year (<span class="text-blue-400 font-bold">{{ draws.length }}</span> draws) data.</p>
        </div>
      </div>

      <!-- Hot & Cold Numbers -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Hot Numbers -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-lg">
          <div class="flex items-center gap-2 mb-4">
            <span class="text-xl">🔥</span>
            <h2 class="font-bold text-red-400">Hot Numbers (Most Frequent)</h2>
          </div>
          <div class="mb-2 text-xs text-gray-500 font-bold uppercase tracking-wider">General Numbers</div>
          <div class="flex justify-between px-2 mb-4 text-white font-mono text-lg">
            <span v-for="n in hotNumbers" :key="n" class="font-bold">{{ n }}</span>
          </div>
        </div>
        <!-- Cold Numbers -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-lg">
          <div class="flex items-center gap-2 mb-4">
            <span class="text-xl">🧊</span>
            <h2 class="font-bold text-blue-400">Cold Numbers (Least Frequent)</h2>
          </div>
          <div class="mb-2 text-xs text-gray-500 font-bold uppercase tracking-wider">General Numbers</div>
          <div class="flex justify-between px-2 mb-4 text-white font-mono text-lg">
             <span v-for="n in coldNumbers" :key="n" class="font-bold text-gray-300">{{ n }}</span>
          </div>
        </div>
      </div>

      <!-- Frequency Chart (Reusing LottoCharts for now as it fits the slot) -->
      <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
        <h2 class="font-bold mb-6 text-white">📈 Number Frequency (Top 20)</h2>
        <div class="h-64 relative w-full overflow-hidden">
           <!-- Using existing Chart component but constraining height -->
           <LottoCharts :draws="draws" :max-num="currentLottery.maxNum" :simple-mode="true" />
        </div>
        <p class="text-center text-xs text-gray-500 mt-4">* Visualizing frequency distribution matrix</p>
      </div>

      <!-- Ratios Row (Placeholder for now) -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
         <!-- Odd/Even -->
         <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
            <h2 class="font-bold mb-4 text-white">🎯 Odd/Even Ratio</h2>
            <div class="h-64 flex items-center justify-center text-gray-500 text-sm">
               Chart Placeholder
            </div>
         </div>
         <!-- Low/High -->
         <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
            <h2 class="font-bold mb-4 text-white">📊 Low/High Ratio</h2>
            <div class="h-64 flex items-center justify-center text-gray-500 text-sm">
               Chart Placeholder
            </div>
         </div>
      </div>
    </div>

    <!-- Tab 2: Generated History (Legacy Style) -->
    <div v-else class="space-y-4">
      <h2 class="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <span>📂</span> Generated History
      </h2>
      
      <div v-if="sortedHistory.length === 0" class="text-center text-gray-500 py-10">
        No history data available.
      </div>

      <div v-else>
         <div v-for="entry in sortedHistory" :key="entry.id" class="glass-panel p-4 rounded-xl border border-white/10 mb-4 bg-[#1e293b]/40">
             <div class="flex justify-between items-start mb-3 border-b border-white/5 pb-2">
                 <div class="text-xs text-gray-400">
                     {{ new Date(entry.date).toLocaleString() }}
                 </div>
                 <div class="flex gap-2">
                     <span class="bg-green-500/20 text-green-300 px-2 py-0.5 rounded text-[10px] font-bold">{{ entry.lotteryName || 'Korea Lotto' }}</span>
                     <span class="bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded text-[10px] font-bold">{{ (entry.model || 'AI').toUpperCase() }}</span>
                 </div>
             </div>
             <div class="space-y-2">
                 <!-- Handle both single array and array of objects format -->
                 <div v-if="Array.isArray(entry.numbers) && entry.numbers.length > 0">
                    <div v-if="typeof entry.numbers[0] === 'number'" class="flex gap-1.5 flex-wrap justify-center sm:justify-start">
                       <span v-for="n in entry.numbers" :key="n" :class="['w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold', getBallColor(n)]">{{ n }}</span>
                    </div>
                    <div v-else v-for="(set, sIdx) in entry.numbers" :key="sIdx" class="flex gap-1.5 flex-wrap justify-center sm:justify-start">
                         <!-- Verify if set is object with numbers array or just array -->
                         <span v-for="n in (set.numbers || set)" :key="n" :class="['w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold', getBallColor(n)]">{{ n }}</span>
                    </div>
                 </div>
             </div>
         </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useLotto } from 'src/composables/useLotto'
import { useHistory } from 'src/composables/useHistory'
import LottoCharts from 'src/components/LottoCharts.vue'
// Chart components would be needed for full charts, skipping for now to strict layout match

const { currentLottery, draws, loadDraws } = useLotto()
const { sortedHistory, loadHistory } = useHistory()

const activeTab = ref('draws')

onMounted(async () => {
  if (draws.value.length === 0) await loadDraws()
  await loadHistory()
})

// Simple Hot/Cold Logic
const numberFreqs = computed(() => {
  const freqs = {}
  draws.value.forEach(d => {
    (d.numbers || []).forEach(n => {
      freqs[n] = (freqs[n] || 0) + 1
    })
  })
  return freqs
})

// Returns top 6 hot numbers
const hotNumbers = computed(() => {
  return Object.entries(numberFreqs.value)
    .sort((a,b) => b[1] - a[1])
    .slice(0, 6)
    .map(x => parseInt(x[0]))
    .sort((a,b) => a-b)
})

// Returns bottom 6 cold numbers (that appeared at least once, or 0)
const coldNumbers = computed(() => {
  const allNums = Array.from({length: 45}, (_, i) => i + 1)
  // Filter for nums with low frequency
  return allNums
    .map(n => ({ n, c: numberFreqs.value[n] || 0 }))
    .sort((a,b) => a.c - b.c)
    .slice(0, 6)
    .map(x => x.n)
    .sort((a,b) => a-b)
})

function getBallColor(n) {
    if (n <= 10) return 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20'
    if (n <= 20) return 'bg-blue-500 text-white shadow-lg shadow-blue-500/20'
    if (n <= 30) return 'bg-red-500 text-white shadow-lg shadow-red-500/20'
    if (n <= 40) return 'bg-gray-600 text-white shadow-lg shadow-gray-500/20'
    return 'bg-green-500 text-white shadow-lg shadow-green-500/20'
}
</script>

<style lang="scss" scoped>
.glass-panel {
  /* Legacy App Style */
}
</style>
