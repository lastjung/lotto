<template>
  <div class="max-w-7xl mx-auto p-4 md:p-8 pb-24 md:pb-8">
    <!-- Tab Navigation -->
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

    <!-- Content Area -->
    <div v-if="activeTab === 'draws'" class="space-y-6">
      <div class="flex items-center justify-between flex-wrap gap-4 mb-4">
        <div>
          <h1 class="text-2xl font-bold mb-1 text-white">📊 Statistics</h1>
          <div class="flex flex-col sm:flex-row gap-2 text-sm text-gray-400">
            <p>최근 <span class="text-blue-400 font-bold">{{ recentDraws.length }}</span>회 데이터 기반 (약 1년)</p>
            <span class="hidden sm:inline text-gray-600">|</span>
            <p class="font-medium text-blue-400">기간: <span class="text-gray-300">{{ periodRange }}</span></p>
          </div>
        </div>

        <!-- Sub Tabs (Mobile Optimized) -->
        <q-btn-toggle
          v-model="activeSubTab"
          flatten
          unelevated
          toggle-color="blue-6"
          color="white"
          text-color="gray-400"
          toggle-text-color="white"
          class="bg-white/5 rounded-xl border border-white/10"
          :options="[
            { value: 'summary', slot: 'summary' },
            { value: 'details', slot: 'details' }
          ]"
        >
          <template v-slot:summary>
            <div class="row items-center no-wrap">
              <q-icon name="dashboard_customize" size="18px" class="q-mr-xs" />
              <div class="text-center font-bold">Summary</div>
            </div>
          </template>
          <template v-slot:details>
            <div class="row items-center no-wrap">
              <q-icon name="analytics" size="18px" class="q-mr-xs" />
              <div class="text-center font-bold">Details</div>
            </div>
          </template>
        </q-btn-toggle>
      </div>

      <!-- Tab 1-1: Summary Content -->
      <div v-if="activeSubTab === 'summary'" class="space-y-6 animate-fade-in">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Hot Numbers -->
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-lg">
            <div class="flex items-center gap-2 mb-4">
              <span class="text-xl">🔥</span>
              <h2 class="font-bold text-red-400">Hot Numbers (최다 출현)</h2>
            </div>
            <div class="mb-4 text-xs text-gray-500 font-bold uppercase tracking-wider flex items-center gap-2">
              General Numbers
              <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
                <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 52회 중 가장 많이 출현한 번호입니다.</q-tooltip>
              </q-icon>
            </div>
            <div class="flex justify-between px-2 mb-4 overflow-x-auto gap-2">
              <div v-for="item in hotNumbers" :key="item.num" class="flex flex-col items-center gap-1">
                <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
                <div class="flex items-center gap-0.5 mt-1">
                  <span class="text-[10px] font-bold text-gray-400">{{ item.count }}</span>
                  <span v-if="item.trend > 0" class="text-[9px] text-red-500 font-bold">▲</span>
                  <span v-else-if="item.trend < 0" class="text-[9px] text-blue-400 font-bold">▼</span>
                </div>
              </div>
            </div>
            <div class="mb-4 text-xs text-gray-500 font-bold uppercase tracking-wider border-t border-white/5 pt-4">Bonus Ball</div>
            <div class="flex justify-start gap-4 px-2">
              <div v-for="item in hotBonus" :key="item.num" class="flex flex-col items-center gap-1">
                <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
                <span class="text-[10px] font-bold text-gray-400">{{ item.count }}</span>
              </div>
            </div>
          </div>

          <!-- Cold Numbers -->
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-lg">
            <div class="flex items-center gap-2 mb-4">
              <span class="text-xl">🧊</span>
              <h2 class="font-bold text-blue-400">Cold Numbers (최소 출현)</h2>
            </div>
            <div class="mb-4 text-xs text-gray-500 font-bold uppercase tracking-wider flex items-center gap-2">
              General Numbers
              <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
                <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 52회 중 가장 적게 출현한 번호입니다.</q-tooltip>
              </q-icon>
            </div>
            <div class="flex justify-between px-2 mb-4 overflow-x-auto gap-2">
              <div v-for="item in coldNumbers" :key="item.num" class="flex flex-col items-center gap-1">
                <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
                <div class="flex items-center gap-0.5 mt-1">
                  <span class="text-[10px] font-bold text-gray-400">{{ item.count }}</span>
                  <span v-if="item.trend > 0" class="text-[9px] text-red-500 font-bold">▲</span>
                  <span v-else-if="item.trend < 0" class="text-[9px] text-blue-400 font-bold">▼</span>
                </div>
              </div>
            </div>
            <div class="mb-4 text-xs text-gray-500 font-bold uppercase tracking-wider border-t border-white/5 pt-4">Bonus Ball</div>
            <div class="flex justify-start gap-4 px-2">
              <div v-for="item in coldBonus" :key="item.num" class="flex flex-col items-center gap-1">
                <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
                <span class="text-[10px] font-bold text-gray-400">{{ item.count }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 rounded-2xl p-6">
            <h2 class="font-bold mb-4 text-white text-sm">🎯 홀짝 및 고저 요약</h2>
            <div class="flex justify-around items-center h-48">
               <canvas ref="oddEvenChartRef"></canvas>
               <canvas ref="lowHighChartRef"></canvas>
            </div>
          </div>
          <div class="glass-panel bg-blue-600/10 border border-blue-500/20 rounded-2xl p-6 flex flex-col justify-center">
            <h3 class="text-blue-400 font-bold text-lg mb-2">분석 가이드 💡</h3>
            <p class="text-gray-300 text-sm leading-relaxed">
              상세 탭에서는 더 많은 통계 그래프를 확인하실 수 있습니다.
            </p>
            <q-btn flat color="blue-4" label="View Detailed Analysis" class="q-mt-md" @click="activeSubTab = 'details'" />
          </div>
        </div>
      </div>

      <!-- Tab 1-2: Details Content -->
      <div v-else class="space-y-6 animate-fade-in">
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-6 text-white text-sm flex items-center gap-2">
            📈 번호별 출현 빈도 (Top 20)
            <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
              <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 52회 기준 상위 20개 번호입니다.</q-tooltip>
            </q-icon>
          </h2>
          <div class="h-64 relative w-full">
            <canvas ref="freqChartRef"></canvas>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
            <h2 class="font-bold mb-4 text-white text-sm flex items-center gap-2">
              📉 합계 추이 (Sum Trend)
              <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
                <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 30회차 합계 변화입니다.</q-tooltip>
              </q-icon>
            </h2>
            <div class="h-64 w-full">
              <canvas ref="sumChartRef"></canvas>
            </div>
          </div>
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
            <h2 class="font-bold mb-4 text-white text-sm flex items-center gap-2">
              🔢 AC값 추이 (AC Value)
              <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
                <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 30회차 AC(복잡도) 추이입니다.</q-tooltip>
              </q-icon>
            </h2>
            <div class="h-64 w-full">
              <canvas ref="acChartRef"></canvas>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
            <h2 class="font-bold mb-4 text-white text-sm flex items-center gap-2">
              📊 합계 분포 (Sum Dist)
              <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
                <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 52회 합계 빈도 분포입니다.</q-tooltip>
              </q-icon>
            </h2>
            <div class="h-64 w-full">
              <canvas ref="sumDistChartRef"></canvas>
            </div>
          </div>
          <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
            <h2 class="font-bold mb-4 text-white text-sm flex items-center gap-2">
              🔢 끝수 분석 (End Digit)
              <q-icon name="info_outline" size="12px" class="cursor-pointer text-gray-600">
                <q-tooltip class="bg-black/90 text-xs shadow-xl border border-white/10">최근 52회 번호의 끝수(0~9) 빈도입니다.</q-tooltip>
              </q-icon>
            </h2>
            <div class="h-64 w-full">
              <canvas ref="endDigitChartRef"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 2: Generated History -->
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
            <div v-if="Array.isArray(entry.numbers) && entry.numbers.length > 0">
              <div v-if="typeof entry.numbers[0] === 'number'" class="flex gap-1.5 flex-wrap justify-center sm:justify-start">
                <span v-for="n in entry.numbers" :key="n" :class="['w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold', getBallColor(n)]">{{ n }}</span>
              </div>
              <div v-else v-for="(set, sIdx) in entry.numbers" :key="sIdx" class="flex gap-1.5 flex-wrap justify-center sm:justify-start">
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
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import { useLotto } from 'src/composables/useLotto'
import { useHistory } from 'src/composables/useHistory'
import { Chart, registerables } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels'

Chart.register(...registerables, ChartDataLabels)

// Set Chart.js defaults for dark mode
Chart.defaults.color = '#94a3b8'
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)'

const { currentLottery, draws, loadDraws } = useLotto()
const { sortedHistory, loadHistory } = useHistory()

const activeTab = ref('draws')
const activeSubTab = ref('summary')

// Chart refs
const freqChartRef = ref(null)
const oddEvenChartRef = ref(null)
const lowHighChartRef = ref(null)
const sumChartRef = ref(null)
const acChartRef = ref(null)
const sumDistChartRef = ref(null)
const endDigitChartRef = ref(null)

// Chart instances
let charts = {}

onMounted(async () => {
  if (draws.value.length === 0) await loadDraws()
  await loadHistory()
  await nextTick()
  renderAllCharts()
})

// Watch for tab changes to re-render charts
watch([activeTab, activeSubTab], async () => {
  if (activeTab.value === 'draws') {
    await nextTick()
    renderAllCharts()
  }
})

// Watch for lottery data changes (deep: true removed for performance)
watch(draws, async () => {
  await nextTick()
  if (activeTab.value === 'draws') {
    renderAllCharts()
  }
}, { deep: false })

// Period range (approx 1 year = 52 weeks)
const recentDraws = computed(() => {
  return draws.value.slice(0, 52)
})

const periodRange = computed(() => {
  if (recentDraws.value.length === 0) return '-'
  const dates = recentDraws.value.map(d => d.draw_date || d.date).filter(Boolean)
  if (dates.length === 0) return '-'
  return `${dates[dates.length - 1]} ~ ${dates[0]}`
})

// Calculate frequencies
const numberFreqs = computed(() => {
  const freqs = {}
  recentDraws.value.forEach(d => {
    (d.numbers || []).forEach(n => freqs[n] = (freqs[n] || 0) + 1)
  })
  return freqs
})

const bonusFreqs = computed(() => {
  const freqs = {}
  recentDraws.value.forEach(d => {
    const bonus = d.bonus_number || d.bonus
    if (bonus) freqs[bonus] = (freqs[bonus] || 0) + 1
  })
  return freqs
})

// Trend calculation (Recent 10 vs Previous 42)
const recent10Count = 10
const previous42Count = 42

const frequenciesRecent10 = computed(() => {
  const freqs = {}
  draws.value.slice(0, recent10Count).forEach(d => {
    (d.numbers || []).forEach(n => freqs[n] = (freqs[n] || 0) + 1)
  })
  return freqs
})

const frequenciesPrevious42 = computed(() => {
  const freqs = {}
  draws.value.slice(recent10Count, recent10Count + previous42Count).forEach(d => {
    (d.numbers || []).forEach(n => freqs[n] = (freqs[n] || 0) + 1)
  })
  return freqs
})

const hotNumbers = computed(() => {
  return Object.entries(numberFreqs.value)
    .map(([num, count]) => {
      const n = parseInt(num)
      const r10 = frequenciesRecent10.value[n] || 0
      const p42 = frequenciesPrevious42.value[n] || 0
      const r10Avg = r10 / recent10Count
      const p42Avg = p42 / previous42Count
      const trend = r10Avg > p42Avg ? 1 : r10Avg < p42Avg ? -1 : 0
      return { num: n, count, trend }
    })
    .sort((a, b) => b.count - a.count)
    .slice(0, 6)
})

const coldNumbers = computed(() => {
  const maxNum = currentLottery.value.maxNum || 45
  const allNums = Array.from({ length: maxNum }, (_, i) => i + 1)
  return allNums
    .map(n => {
      const count = numberFreqs.value[n] || 0
      const r10 = frequenciesRecent10.value[n] || 0
      const p42 = frequenciesPrevious42.value[n] || 0
      const r10Avg = r10 / recent10Count
      const p42Avg = p42 / previous42Count
      const trend = r10Avg > p42Avg ? 1 : r10Avg < p42Avg ? -1 : 0
      return { num: n, count, trend }
    })
    .sort((a, b) => a.count - b.count)
    .slice(0, 6)
})

const hotBonus = computed(() => {
  return Object.entries(bonusFreqs.value)
    .map(([num, count]) => ({ num: parseInt(num), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 3)
})

const coldBonus = computed(() => {
  const maxNum = currentLottery.value.maxNum || 45
  const allBonusNums = []
  for (let n = 1; n <= maxNum; n++) {
    allBonusNums.push({ num: n, count: bonusFreqs.value[n] || 0 })
  }
  return allBonusNums
    .sort((a, b) => a.count - b.count)
    .slice(0, 3)
})

// Styled ball helpers
function getBallColorClass(n) {
  if (n <= 10) return 'bg-ball-yellow'
  if (n <= 20) return 'bg-ball-blue'
  if (n <= 30) return 'bg-ball-red'
  if (n <= 40) return 'bg-ball-gray'
  return 'bg-ball-green'
}

function getBallColor(n) {
  if (n <= 10) return 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20'
  if (n <= 20) return 'bg-blue-500 text-white shadow-lg shadow-blue-500/20'
  if (n <= 30) return 'bg-red-500 text-white shadow-lg shadow-red-500/20'
  if (n <= 40) return 'bg-gray-600 text-white shadow-lg shadow-gray-500/20'
  return 'bg-green-500 text-white shadow-lg shadow-green-500/20'
}

// AC Value calculation: Set of (n[i]-n[j]) size - (count-1)
function calculateAC(numbers) {
  if (!numbers || numbers.length < 2) return 0
  const diffs = new Set()
  for (let i = 0; i < numbers.length; i++) {
    for (let j = i + 1; j < numbers.length; j++) {
      diffs.add(Math.abs(numbers[i] - numbers[j]))
    }
  }
  return diffs.size - (numbers.length - 1)
}

function renderAllCharts() {
  if (recentDraws.value.length === 0) return

  // Freq Top 20
  const top20 = Object.entries(numberFreqs.value)
    .map(([num, count]) => ({ num: parseInt(num), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20)
  renderFreqChart(top20)

  // Odd/Even
  let odd = 0, even = 0
  recentDraws.value.forEach(d => (d.numbers || []).forEach(n => n % 2 === 0 ? even++ : odd++))
  renderOddEvenChart(odd, even)

  // Low/High
  const midPoint = (currentLottery.value.maxNum || 45) / 2
  let lowCount = 0, highCount = 0
  recentDraws.value.forEach(d => (d.numbers || []).forEach(n => n <= midPoint ? lowCount++ : highCount++))
  renderLowHighChart(lowCount, highCount)

  // Trends & Distributions (Details)
  if (activeSubTab.value === 'details') {
    const last30 = recentDraws.value.slice(0, 30).reverse()
    const sums = last30.map(d => (d.numbers || []).reduce((a, b) => a + b, 0))
    const labels = last30.map(d => d.draw_no || d.draw_number || '')
    renderSumChart(labels, sums)
    renderACChart(labels, last30.map(d => calculateAC(d.numbers || [])))

    // Sum Distribution
    const allSums = recentDraws.value.map(d => (d.numbers || []).reduce((a, b) => a + b, 0))
    if (allSums.length > 0) {
      const binSize = 10
      const minSum = Math.min(...allSums)
      const maxSum = Math.max(...allSums)
      const startBin = Math.floor(minSum / binSize) * binSize
      const endBin = Math.ceil(maxSum / binSize) * binSize
      const labelsDist = []
      const dataDist = []
      for (let i = startBin; i < endBin; i += binSize) {
        labelsDist.push(`${i}~${i + binSize - 1}`)
        dataDist.push(allSums.filter(s => s >= i && s < i + binSize).length)
      }
      renderSumDistChart(labelsDist, dataDist)
    }

    // End Digit
    const endDigitFreq = new Array(10).fill(0)
    recentDraws.value.forEach(d => (d.numbers || []).forEach(n => endDigitFreq[n % 10]++))
    renderEndDigitChart(endDigitFreq)
  }
}

// Chart renderers (Simplified)
function renderFreqChart(data) {
  if (!freqChartRef.value) return
  const ctx = freqChartRef.value.getContext('2d')
  if (charts.freq) charts.freq.destroy()
  charts.freq = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.num),
      datasets: [{ label: '회수', data: data.map(d => d.count), backgroundColor: '#6366f1', borderRadius: 4 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, datalabels: { display: false } },
      scales: { y: { beginAtZero: true }, x: { grid: { display: false } } }
    }
  })
}

function renderOddEvenChart(odd, even) {
  if (!oddEvenChartRef.value) return
  const ctx = oddEvenChartRef.value.getContext('2d')
  if (charts.oddEven) charts.oddEven.destroy()
  charts.oddEven = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['홀', '짝'],
      datasets: [{ data: [odd, even], backgroundColor: ['#818cf8', '#34d399'], borderWidth: 0 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } },
        datalabels: { color: 'white', font: { weight: 'bold', size: 12 }, formatter: (v) => ((v/(odd+even))*100).toFixed(0) + '%' }
      }
    }
  })
}

function renderLowHighChart(low, high) {
  if (!lowHighChartRef.value) return
  const ctx = lowHighChartRef.value.getContext('2d')
  if (charts.lowHigh) charts.lowHigh.destroy()
  charts.lowHigh = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['고', '저'],
      datasets: [{ data: [high, low], backgroundColor: ['#f87171', '#60a5fa'], borderWidth: 0 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } },
        datalabels: { color: 'white', font: { weight: 'bold', size: 12 }, formatter: (v) => ((v/(low+high))*100).toFixed(0) + '%' }
      }
    }
  })
}

function renderSumChart(labels, data) {
  if (!sumChartRef.value) return
  const ctx = sumChartRef.value.getContext('2d')
  if (charts.sum) charts.sum.destroy()
  charts.sum = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ data, borderColor: '#34d399', backgroundColor: 'rgba(52, 211, 153, 0.1)', fill: true, tension: 0.4 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, datalabels: { display: false } } }
  })
}

function renderACChart(labels, data) {
  if (!acChartRef.value) return
  const ctx = acChartRef.value.getContext('2d')
  if (charts.ac) charts.ac.destroy()
  charts.ac = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ data, borderColor: '#ec4899', backgroundColor: 'rgba(236, 72, 153, 0.1)', tension: 0.1 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, datalabels: { display: false } } }
  })
}

function renderSumDistChart(labels, data) {
  if (!sumDistChartRef.value) return
  const ctx = sumDistChartRef.value.getContext('2d')
  if (charts.sumDist) charts.sumDist.destroy()
  charts.sumDist = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: '#a78bfa' }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, datalabels: { display: false } } }
  })
}

function renderEndDigitChart(data) {
  if (!endDigitChartRef.value) return
  const ctx = endDigitChartRef.value.getContext('2d')
  if (charts.endDigit) charts.endDigit.destroy()
  charts.endDigit = new Chart(ctx, {
    type: 'bar',
    data: { labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], datasets: [{ data, backgroundColor: '#fbbf24' }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, datalabels: { display: false } } }
  })
}
</script>

<style lang="scss" scoped>
.glass-panel {
  background: rgba(30, 41, 59, 0.5);
  backdrop-filter: blur(12px);
}

.ball {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-weight: bold;
  font-size: 14px;
  color: white;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), inset 0 -2px 4px rgba(0, 0, 0, 0.2);
}

.bg-ball-yellow { background-color: #eab308; color: #000; }
.bg-ball-blue { background-color: #3b82f6; }
.bg-ball-red { background-color: #ef4444; }
.bg-ball-gray { background-color: #6b7280; }
.bg-ball-green { background-color: #22c55e; }

.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
