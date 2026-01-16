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

    <!-- Tab 1: Draws Analysis -->
    <div v-if="activeTab === 'draws'" class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold mb-1 text-white">📊 Statistics</h1>
        <div class="flex flex-col sm:flex-row gap-2 text-sm text-gray-400">
          <p>최근 <span class="text-blue-400 font-bold">{{ recentDraws.length }}</span>회 데이터 기반 (약 1년)</p>
          <span class="hidden sm:inline text-gray-600">|</span>
          <p class="font-medium text-blue-400">기간: <span class="text-gray-300">{{ periodRange }}</span></p>
        </div>
      </div>

      <!-- Hot & Cold Numbers -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Hot Numbers -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-lg">
          <div class="flex items-center gap-2 mb-4">
            <span class="text-xl">🔥</span>
            <h2 class="font-bold text-red-400">Hot Numbers (최다 출현)</h2>
          </div>
          <div class="mb-2 text-xs text-gray-500 font-bold uppercase tracking-wider">General Numbers</div>
          <div class="flex justify-between px-2 mb-4">
            <div v-for="item in hotNumbers" :key="item.num" class="flex flex-col items-center gap-1">
              <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
              <span class="text-xs font-bold text-gray-400">{{ item.count }}회</span>
            </div>
          </div>
          <div class="mb-2 text-xs text-gray-500 font-bold uppercase tracking-wider border-t border-white/5 pt-4">Bonus Ball</div>
          <div class="flex justify-start gap-4 px-2">
            <div v-for="item in hotBonus" :key="item.num" class="flex flex-col items-center gap-1">
              <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
              <span class="text-xs font-bold text-gray-400">{{ item.count }}회</span>
            </div>
            <span v-if="hotBonus.length === 0" class="text-xs text-gray-500">데이터 없음</span>
          </div>
        </div>
        <!-- Cold Numbers -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-lg">
          <div class="flex items-center gap-2 mb-4">
            <span class="text-xl">🧊</span>
            <h2 class="font-bold text-blue-400">Cold Numbers (최소 출현)</h2>
          </div>
          <div class="mb-2 text-xs text-gray-500 font-bold uppercase tracking-wider">General Numbers</div>
          <div class="flex justify-between px-2 mb-4">
            <div v-for="item in coldNumbers" :key="item.num" class="flex flex-col items-center gap-1">
              <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
              <span class="text-xs font-bold text-gray-400">{{ item.count }}회</span>
            </div>
          </div>
          <div class="mb-2 text-xs text-gray-500 font-bold uppercase tracking-wider border-t border-white/5 pt-4">Bonus Ball</div>
          <div class="flex justify-start gap-4 px-2">
            <div v-for="item in coldBonus" :key="item.num" class="flex flex-col items-center gap-1">
              <div :class="['ball', getBallColorClass(item.num)]">{{ item.num }}</div>
              <span class="text-xs font-bold text-gray-400">{{ item.count }}회</span>
            </div>
            <span v-if="coldBonus.length === 0" class="text-xs text-gray-500">데이터 없음</span>
          </div>
        </div>
      </div>

      <!-- Frequency Chart -->
      <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
        <h2 class="font-bold mb-6 text-white">📈 번호별 출현 빈도 (Top 20)</h2>
        <div class="h-64 relative w-full">
          <canvas ref="freqChartRef"></canvas>
        </div>
        <p class="text-center text-xs text-gray-500 mt-4">* 상위 20개 번호만 표시됩니다.</p>
      </div>

      <!-- Ratios Row -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Odd/Even -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-4 text-white">🎯 홀짝 비율 (Odd/Even)</h2>
          <div class="h-64 flex items-center justify-center">
            <canvas ref="oddEvenChartRef"></canvas>
          </div>
        </div>
        <!-- Low/High -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-4 text-white">📊 고저 비율 (Low/High)</h2>
          <div class="h-64 flex items-center justify-center">
            <canvas ref="lowHighChartRef"></canvas>
          </div>
        </div>
      </div>

      <!-- Trends Row -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Sum Trend -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-4 text-white">📉 합계 추이 (Sum Trend)</h2>
          <div class="h-64 w-full">
            <canvas ref="sumChartRef"></canvas>
          </div>
          <p class="text-center text-xs text-gray-500 mt-2">* 최근 30회차 데이터</p>
        </div>
        <!-- AC Trend -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-4 text-white">🔢 AC값 추이 (AC Value)</h2>
          <div class="h-64 w-full">
            <canvas ref="acChartRef"></canvas>
          </div>
          <p class="text-center text-xs text-gray-500 mt-2">* 산술적 복잡도 (숫자 간격의 불규칙성)</p>
        </div>
      </div>

      <!-- Distributions Row -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Sum Distribution -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-4 text-white">📊 합계 분포 (Sum Dist)</h2>
          <div class="h-64 w-full">
            <canvas ref="sumDistChartRef"></canvas>
          </div>
          <p class="text-center text-xs text-gray-500 mt-2">* 전체 기간 합계 빈도 (정규분포)</p>
        </div>
        <!-- End Digit -->
        <div class="glass-panel bg-[#1e293b]/50 border border-white/5 backdrop-blur-md rounded-2xl p-6 shadow-xl">
          <h2 class="font-bold mb-4 text-white">🔢 끝수 분석 (End Digit)</h2>
          <div class="h-64 w-full">
            <canvas ref="endDigitChartRef"></canvas>
          </div>
          <p class="text-center text-xs text-gray-500 mt-2">* 번호의 1의 자리 (0~9) 출현 빈도</p>
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
watch(activeTab, async (newTab) => {
  if (newTab === 'draws') {
    await nextTick()
    renderAllCharts()
  }
})

// Watch for lottery data changes to re-render charts
watch(draws, async () => {
  await nextTick()
  if (activeTab.value === 'draws') {
    renderAllCharts()
  }
}, { deep: false })

// Period range
// Limit to recent 52 draws (approx 1 year) like web version
const recentDraws = computed(() => {
  return draws.value.slice(0, 52)
})

// Period range (based on recent draws)
const periodRange = computed(() => {
  if (recentDraws.value.length === 0) return '-'
  const dates = recentDraws.value.map(d => d.draw_date || d.date).filter(Boolean)
  if (dates.length === 0) return '-'
  return `${dates[dates.length - 1]} ~ ${dates[0]}`
})

// Calculate frequencies (based on recent 52 draws)
const numberFreqs = computed(() => {
  const freqs = {}
  recentDraws.value.forEach(d => {
    (d.numbers || []).forEach(n => {
      freqs[n] = (freqs[n] || 0) + 1
    })
  })
  return freqs
})

const bonusFreqs = computed(() => {
  const freqs = {}
  recentDraws.value.forEach(d => {
    const bonus = d.bonus_number || d.bonus
    if (bonus) {
      freqs[bonus] = (freqs[bonus] || 0) + 1
    }
  })
  return freqs
})

// Hot/Cold calculations
const hotNumbers = computed(() => {
  return Object.entries(numberFreqs.value)
    .map(([num, count]) => ({ num: parseInt(num), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6)
})

const coldNumbers = computed(() => {
  const allNums = Array.from({ length: currentLottery.value.maxNum || 45 }, (_, i) => i + 1)
  return allNums
    .map(n => ({ num: n, count: numberFreqs.value[n] || 0 }))
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
  // Include ALL possible bonus numbers (1 to maxNum), even those with 0 appearances
  const maxNum = currentLottery.value.maxNum || 45
  const allBonusNums = []
  
  for (let n = 1; n <= maxNum; n++) {
    allBonusNums.push({ num: n, count: bonusFreqs.value[n] || 0 })
  }
  
  // Sort by frequency ascending (least frequent first, including 0)
  return allBonusNums
    .sort((a, b) => a.count - b.count)
    .slice(0, 3)
})

// Ball color class for styled balls
function getBallColorClass(n) {
  if (n <= 10) return 'bg-ball-yellow'
  if (n <= 20) return 'bg-ball-blue'
  if (n <= 30) return 'bg-ball-red'
  if (n <= 40) return 'bg-ball-gray'
  return 'bg-ball-green'
}

// Ball color for history list
function getBallColor(n) {
  if (n <= 10) return 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20'
  if (n <= 20) return 'bg-blue-500 text-white shadow-lg shadow-blue-500/20'
  if (n <= 30) return 'bg-red-500 text-white shadow-lg shadow-red-500/20'
  if (n <= 40) return 'bg-gray-600 text-white shadow-lg shadow-gray-500/20'
  return 'bg-green-500 text-white shadow-lg shadow-green-500/20'
}

// AC Value calculation
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

// Render all charts (using recentDraws for consistency with period display)
function renderAllCharts() {
  if (recentDraws.value.length === 0) return

  // Frequency Chart (Top 20) - already uses numberFreqs which is based on recentDraws
  const top20 = Object.entries(numberFreqs.value)
    .map(([num, count]) => ({ num: parseInt(num), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20)
  renderFreqChart(top20)

  // Odd/Even (based on recentDraws)
  let odd = 0, even = 0
  recentDraws.value.forEach(d => (d.numbers || []).forEach(n => n % 2 === 0 ? even++ : odd++))
  renderOddEvenChart(odd, even)

  // Low/High (based on recentDraws)
  const midPoint = (currentLottery.value.maxNum || 45) / 2
  let lowCount = 0, highCount = 0
  recentDraws.value.forEach(d => (d.numbers || []).forEach(n => n <= midPoint ? lowCount++ : highCount++))
  renderLowHighChart(lowCount, highCount)

  // Sum Trend (Last 30 from recentDraws)
  const last30 = recentDraws.value.slice(0, 30).reverse()
  const sums = last30.map(d => (d.numbers || []).reduce((a, b) => a + b, 0))
  const labels = last30.map(d => d.draw_no || d.draw_number || '')
  renderSumChart(labels, sums)

  // AC Trend (Last 30 from recentDraws)
  const acValues = last30.map(d => calculateAC(d.numbers || []))
  renderACChart(labels, acValues)

  // Sum Distribution (based on recentDraws)
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

  // End Digit (based on recentDraws)
  const endDigitFreq = new Array(10).fill(0)
  recentDraws.value.forEach(d => (d.numbers || []).forEach(n => endDigitFreq[n % 10]++))
  renderEndDigitChart(endDigitFreq)
}

// Chart renderers
function renderFreqChart(data) {
  if (!freqChartRef.value) return
  const ctx = freqChartRef.value.getContext('2d')
  if (charts.freq) charts.freq.destroy()
  charts.freq = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.num),
      datasets: [{
        label: '출현 횟수',
        data: data.map(d => d.count),
        backgroundColor: '#6366f1',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, datalabels: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { borderDash: [2, 2] } },
        x: { grid: { display: false } }
      }
    }
  })
}

function renderOddEvenChart(odd, even) {
  if (!oddEvenChartRef.value) return
  const ctx = oddEvenChartRef.value.getContext('2d')
  const total = odd + even
  if (charts.oddEven) charts.oddEven.destroy()
  charts.oddEven = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Odd (홀)', 'Even (짝)'],
      datasets: [{
        data: [odd, even],
        backgroundColor: ['#818cf8', '#34d399'],
        borderWidth: 0
      }]
    },
    options: {
      rotation: -90,
      circumference: 360,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true } },
        datalabels: {
          color: 'white',
          font: { weight: 'bold', size: 14 },
          formatter: (value) => total === 0 ? '0%' : ((value / total) * 100).toFixed(1) + '%'
        }
      }
    }
  })
}

function renderLowHighChart(low, high) {
  if (!lowHighChartRef.value) return
  const ctx = lowHighChartRef.value.getContext('2d')
  const total = low + high
  if (charts.lowHigh) charts.lowHigh.destroy()
  charts.lowHigh = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['High (고)', 'Low (저)'],
      datasets: [{
        data: [high, low],
        backgroundColor: ['#f87171', '#60a5fa'],
        borderWidth: 0
      }]
    },
    options: {
      rotation: -90,
      circumference: 360,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true } },
        datalabels: {
          color: 'white',
          font: { weight: 'bold', size: 14 },
          formatter: (value) => total === 0 ? '0%' : ((value / total) * 100).toFixed(1) + '%'
        }
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
    data: {
      labels,
      datasets: [{
        label: '합계',
        data,
        borderColor: '#34d399',
        backgroundColor: 'rgba(52, 211, 153, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: 'white',
        pointBorderColor: '#34d399',
        pointBorderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, datalabels: { display: false } },
      scales: {
        y: { grid: { borderDash: [2, 2] } },
        x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }
      }
    }
  })
}

function renderACChart(labels, data) {
  if (!acChartRef.value) return
  const ctx = acChartRef.value.getContext('2d')
  if (charts.ac) charts.ac.destroy()
  charts.ac = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'AC Value',
        data,
        borderColor: '#ec4899',
        backgroundColor: 'rgba(236, 72, 153, 0.1)',
        borderWidth: 2,
        tension: 0.1,
        pointRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, datalabels: { display: false } },
      scales: {
        y: { beginAtZero: false, suggestedMin: 0, suggestedMax: 10, grid: { borderDash: [2, 2] } },
        x: { display: false }
      }
    }
  })
}

function renderSumDistChart(labels, data) {
  if (!sumDistChartRef.value) return
  const ctx = sumDistChartRef.value.getContext('2d')
  if (charts.sumDist) charts.sumDist.destroy()
  charts.sumDist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: '빈도수',
        data,
        backgroundColor: '#a78bfa',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, datalabels: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { borderDash: [2, 2] } },
        x: { grid: { display: false } }
      }
    }
  })
}

function renderEndDigitChart(data) {
  if (!endDigitChartRef.value) return
  const ctx = endDigitChartRef.value.getContext('2d')
  if (charts.endDigit) charts.endDigit.destroy()
  charts.endDigit = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      datasets: [{
        label: '출현 빈도',
        data,
        backgroundColor: '#fbbf24',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, datalabels: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { borderDash: [2, 2] } },
        x: { grid: { display: false } }
      }
    }
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
</style>
