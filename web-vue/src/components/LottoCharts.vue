<template>
  <div class="row q-col-gutter-lg">
    <!-- Hot Numbers Chart -->
    <div class="col-12 col-lg-6">
      <div class="glass-panel p-6 bg-[#1e293b]/50 rounded-2xl h-64 relative overflow-hidden group">
        <div class="flex justify-between items-start z-10 relative">
          <div>
            <h3 class="font-bold text-white text-sm tracking-wider flex items-center gap-2">
              <div class="w-1 h-4 bg-green-500 rounded-full"></div>
              HOT NUMBERS (LAST 50)
            </h3>
            <p class="text-[10px] text-gray-500 mt-1 uppercase">Frequency distribution</p>
          </div>
          <span class="text-[10px] font-mono text-green-400 bg-green-900/20 px-2 py-1 rounded border border-green-500/20">LIVE DATA ACTIVE</span>
        </div>
        
        <div class="flex items-end justify-between gap-1 mt-6 px-2 h-32 relative z-10">
          <div 
            v-for="(freq, i) in hotNumbers" 
            :key="i"
            class="bar-wrapper group/bar"
            :style="{ height: `${(freq / maxFreq) * 100}%` }"
          >
            <div class="bar bg-gradient-to-t from-green-600 to-green-400 rounded-t-sm w-full h-full relative">
               <div class="bar-glow"></div>
               <div class="tooltip opacity-0 group-hover/bar:opacity-100 transition-opacity">
                  {{ i + 1 }}: {{ freq }}
               </div>
            </div>
          </div>
        </div>
        
        <div class="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none"></div>
      </div>
    </div>

    <!-- Sum Distribution Chart -->
    <div class="col-12 col-lg-6">
      <div class="glass-panel p-6 bg-[#1e293b]/50 rounded-2xl h-64 relative overflow-hidden">
        <div class="flex justify-between items-start z-10 relative">
          <div>
            <h3 class="font-bold text-white text-sm tracking-wider flex items-center gap-2">
              <div class="w-1 h-4 bg-purple-500 rounded-full"></div>
              SUM DISTRIBUTION THEORY
            </h3>
            <p class="text-[10px] text-gray-500 mt-1 uppercase">Gaussian Normal Distribution</p>
          </div>
          <div class="z-20 flex gap-2">
            <span class="text-[10px] text-gray-400 bg-black/20 px-2 py-1 rounded border border-white/5">100-175 OPTIMAL</span>
          </div>
        </div>

        <div class="mt-6 relative w-full h-32 z-10">
          <svg class="w-full h-full" overflow="visible" preserveAspectRatio="none" viewBox="0 0 100 100">
            <!-- Background Area -->
            <path d="M 0 100 Q 20 100 35 70 T 50 20 T 65 70 T 100 100 L 100 100 L 0 100 Z" fill="url(#purple-gradient)" opacity="0.1" />
            <!-- Theoretical Curve -->
            <path d="M 0 100 Q 20 100 35 70 T 50 20 T 65 70 T 100 100" fill="none" stroke="#a855f7" stroke-width="2" stroke-dasharray="200" stroke-dashoffset="0" class="curve-anim" />
            <defs>
              <linearGradient id="purple-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#a855f7;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#a855f7;stop-opacity:0" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        <div class="absolute bottom-4 left-0 w-full px-6 flex justify-between text-[10px] text-gray-600 font-mono">
            <span>LOW SUM</span>
            <span class="text-purple-400 font-bold">OPTIMAL ZONE</span>
            <span>HIGH SUM</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  draws: {
    type: Array,
    default: () => []
  }
})

// Generate mock hot numbers if no real data
const hotNumbers = computed(() => {
  if (props.draws.length === 0) {
    return Array.from({ length: 45 }, () => Math.floor(Math.random() * 15))
  }
  // Simplified frequency count for demo
  const freqs = Array(45).fill(0)
  props.draws.forEach(d => {
    (d.numbers || []).forEach(n => {
      if (n >= 1 && n <= 45) freqs[n-1]++
    })
  })
  return freqs
})

const maxFreq = computed(() => Math.max(...hotNumbers.value, 1))
</script>

<style lang="scss" scoped>
.bar-wrapper {
  flex: 1;
  min-width: 4px;
  position: relative;
}

.bar {
  transition: all 0.5s ease;
  &:hover {
    filter: brightness(1.3);
  }
}

.bar-glow {
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 150%;
  height: 20px;
  background: radial-gradient(circle, rgba(74, 222, 128, 0.3) 0%, transparent 70%);
  filter: blur(4px);
  opacity: 0;
  transition: opacity 0.3s;
}

.group\/bar:hover .bar-glow {
  opacity: 1;
}

.tooltip {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%) translateY(-5px);
  background: #000;
  color: #fff;
  padding: 2px 4px;
  border-radius: 4px;
  font-size: 8px;
  white-space: nowrap;
  pointer-events: none;
  z-index: 50;
  border: 1px solid rgba(255,255,255,0.1);
}

.curve-anim {
  animation: draw 2s ease-out forwards;
}

@keyframes draw {
  from { stroke-dashoffset: 200; }
  to { stroke-dashoffset: 0; }
}
</style>
