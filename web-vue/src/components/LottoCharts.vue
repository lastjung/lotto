<template>
  <div :class="simpleMode ? 'w-full h-full' : 'row q-col-gutter-lg'">
    <!-- Hot Numbers Chart -->
    <div :class="simpleMode ? 'w-full h-full' : 'col-12 col-lg-6'">
      <div 
        class="glass-panel p-6 bg-[#1e293b]/40 border border-white/5 rounded-2xl relative overflow-hidden group"
        :class="simpleMode ? 'h-full border-none bg-transparent !p-0' : 'h-64'"
      >
        <div class="flex justify-between items-start z-10 relative">
          <div v-if="!simpleMode">
            <h3 class="font-bold text-white text-[12px] tracking-[0.2em] flex items-center gap-2 uppercase">
              <div class="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_10px_#22c55e]"></div>
              Hot Numbers (Last 52)
            </h3>
            <p class="text-[9px] text-gray-500 mt-1 uppercase tracking-widest pl-3.5">Top Frequent Numbers</p>
          </div>
          
          <span v-if="!simpleMode" class="text-[9px] font-bold text-green-400 bg-green-900/20 px-2 py-1 rounded border border-green-500/20 tracking-tighter shadow-sm">LIVE FEED ACTIVE</span>
        </div>
        
        <div class="flex items-end justify-around gap-2 px-1 relative z-10" :class="simpleMode ? 'h-full mt-0' : 'h-32 mt-8'">
          <div 
            v-for="item in hotNumbers" 
            :key="item.num"
            class="flex flex-col items-center gap-2 group/bar flex-1 h-full justify-end"
          >
            <div 
                class="bar bg-gradient-to-t from-green-600/80 to-green-400 rounded-t-lg w-full max-w-[32px] relative transition-all duration-500 hover:brightness-125 hover:scale-x-110 z-10 flex items-end justify-center"
                :style="{ height: `${(item.freq / Math.max(maxFreq, 1)) * 100}%` }"
            >
               <span class="text-[10px] font-bold text-black/50 mb-1">{{ item.freq }}</span>
               <!-- Tooltip -->
               <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-[8px] rounded opacity-0 group-hover/bar:opacity-100 transition-opacity z-50 pointer-events-none whitespace-nowrap border border-white/10 shadow-xl">
                  Number #{{ item.num }}: {{ item.freq }} times
               </div>
            </div>
            <span class="text-xs font-bold text-white">{{ item.num }}</span>
          </div>
        </div>
        
        <div v-if="!simpleMode" class="absolute top-0 right-0 w-48 h-48 bg-green-500/5 rounded-full blur-[80px] -mr-16 -mt-16 pointer-events-none"></div>
      </div>
    </div>

    <!-- Sum Distribution Chart -->
    <div v-if="!simpleMode" class="col-12 col-lg-6">
      <div class="glass-panel p-6 bg-[#1e293b]/40 border border-white/5 rounded-2xl h-64 relative overflow-hidden group">
        <div class="flex justify-between items-start z-10 relative">
          <div>
            <h3 class="font-bold text-white text-[12px] tracking-[0.2em] flex items-center gap-2 uppercase">
              <div class="w-1.5 h-1.5 rounded-full bg-purple-500 shadow-[0_0_10px_#a855f7]"></div>
              Sum Theory Curve
            </h3>
            <p class="text-[9px] text-gray-500 mt-1 uppercase tracking-widest pl-3.5">Gaussian Normal Distribution</p>
          </div>
          <div class="z-20 flex gap-2">
            <span class="text-[9px] text-gray-400 bg-black/40 px-2 py-1 rounded border border-white/5 font-bold uppercase tracking-widest">100-175 Optimal</span>
          </div>
        </div>

        <div class="mt-8 relative w-full h-32 z-10 flex items-end">
          <svg class="w-full h-full" overflow="visible" preserveAspectRatio="none" viewBox="0 0 100 100">
            <path d="M 0 100 C 10 95, 20 85, 30 70 S 45 10, 50 10 S 65 10, 70 70 S 90 95, 100 100 L 100 100 L 0 100 Z" fill="url(#purple-gradient-v4)" opacity="0.3" />
            <path d="M 0 100 C 10 95, 20 85, 30 70 S 45 10, 50 10 S 65 10, 70 70 S 90 95, 100 100" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" class="curve-glow" />
            <defs>
              <linearGradient id="purple-gradient-v4" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#a855f7;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#0f172a;stop-opacity:0" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        <div class="absolute bottom-4 left-0 w-full px-6 flex justify-between text-[9px] text-gray-500 font-bold tracking-[0.2em] uppercase opacity-40">
            <span>Low Range</span>
            <span class="text-blue-400 tracking-[0.4em]">Optimal Quantum Zone</span>
            <span>High Range</span>
        </div>
        
        <div class="absolute top-0 right-0 w-48 h-48 bg-purple-500/5 rounded-full blur-[80px] -mr-16 -mt-16 pointer-events-none"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  draws: { type: Array, default: () => [] },
  maxNum: { type: Number, default: 45 },
  simpleMode: { type: Boolean, default: false }
})

const hotNumbers = computed(() => {
  const max = props.maxNum || 45
  const freqs = Array(max).fill(0)
  
  if (props.draws.length > 0) {
    props.draws.forEach(d => {
      (d.numbers || []).forEach(n => { if (n >= 1 && n <= max) freqs[n-1]++ })
    })
  }

  // Create array of {num, freq}, sort by freq desc, take top 6
  return freqs
    .map((f, i) => ({ num: i + 1, freq: f }))
    .sort((a, b) => b.freq - a.freq)
    .slice(0, 6)
})

const maxFreq = computed(() => {
  if (hotNumbers.value.length === 0) return 1
  return Math.max(...hotNumbers.value.map(item => item.freq), 1)
})
</script>

<style lang="scss" scoped>
.curve-glow {
  filter: drop-shadow(0 0 5px rgba(168, 85, 247, 0.5));
}
.bar-wrapper:hover {
  z-index: 50;
}
</style>
