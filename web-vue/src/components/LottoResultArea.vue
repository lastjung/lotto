<template>
  <div class="lotto-result-area glass-panel p-10 relative overflow-hidden group">
    <!-- Scanner Glow Effect -->
    <div v-if="scanning" class="scanner-line"></div>
    
    <!-- Background Decoration -->
    <div class="absolute -top-24 -right-24 w-64 h-64 bg-primary/10 rounded-full blur-3xl pointer-events-none group-hover:bg-primary/20 transition-colors"></div>
    <div class="absolute -bottom-24 -left-24 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-blue-500/20 transition-colors"></div>

    <div class="flex flex-col items-center justify-center min-h-[450px] relative z-10">
      <div v-if="!results && !generating" class="text-center">
        <div class="orb-container mb-10" @click="$emit('generate')">
          <div class="main-orb pulsing-core"></div>
          <div class="orb-glow"></div>
          <span class="orb-icon">🔮</span>
        </div>
        <h2 class="text-3xl font-bold text-white mb-3 heading-font tracking-tight">Ready for Analysis</h2>
        <p class="text-gray-400 max-w-xs mx-auto mb-10 text-sm leading-relaxed">Select a strategy on the left and start the Multi-Layer Analysis.</p>
      </div>

      <div v-else class="w-full">
        <!-- Result Header -->
        <div class="flex justify-between items-center mb-10 border-b border-white/5 pb-6">
          <div class="flex items-center gap-3">
             <div class="w-3 h-3 rounded-full bg-pink-500 shadow-glow-pink animate-pulse"></div>
             <h3 class="font-bold text-white tracking-widest uppercase text-sm">Generation Complete</h3>
          </div>
          <div class="flex gap-2">
            <span class="text-[10px] bg-blue-500/20 text-blue-400 px-3 py-1.5 rounded-full border border-blue-500/30 font-bold tracking-wider uppercase">
               {{ modelName }}
            </span>
          </div>
        </div>

        <!-- Balls Container -->
        <div class="flex justify-center gap-4 mb-16 flex-wrap">
          <LottoBall 
            v-for="(num, i) in results" 
            :key="i"
            :number="num"
            size="72px"
            class="pop-animation shadow-2xl"
            :style="{ animationDelay: `${i * 0.1}s` }"
          />
        </div>

        <!-- Analysis Grid -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
           <div class="stat-card group">
              <span class="text-[10px] text-gray-500 uppercase tracking-widest block mb-1">Sum Total</span>
              <div class="text-xl font-bold text-white group-hover:text-primary transition-colors">{{ analysis?.sum || '-' }}</div>
           </div>
           <div class="stat-card group">
              <span class="text-[10px] text-gray-500 uppercase tracking-widest block mb-1">AC Value</span>
              <div class="text-xl font-bold text-white group-hover:text-primary transition-colors">{{ analysis?.ac_value || '-' }}</div>
           </div>
           <div class="stat-card group">
              <span class="text-[10px] text-gray-500 uppercase tracking-widest block mb-1">Odd:Even</span>
              <div class="text-xl font-bold text-white group-hover:text-primary transition-colors">{{ analysis?.odd_even || '-' }}</div>
           </div>
           <div class="stat-card group border-primary/20">
              <span class="text-[10px] text-primary/70 uppercase tracking-widest block mb-1">Confidence</span>
              <div class="text-xl font-bold text-green-400">98.4%</div>
           </div>
        </div>

        <div class="mt-12 text-center">
           <q-btn
              flat
              rounded
              color="gray-4"
              icon="refresh"
              label="Re-analyze Quantum Space"
              class="text-xs uppercase tracking-widest opacity-60 hover:opacity-100"
              @click="$emit('generate')"
           />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import LottoBall from './LottoBall.vue'

defineProps({
  results: Array,
  analysis: Object,
  generating: Boolean,
  scanning: Boolean,
  modelName: String
})

defineEmits(['generate'])
</script>

<style lang="scss" scoped>
.lotto-result-area {
  background: rgba(15, 23, 42, 0.4);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 40px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

.scanner-line {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, transparent, #a855f7, transparent);
  box-shadow: 0 0 20px #a855f7;
  z-index: 20;
  animation: scan 3s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}

@keyframes scan {
  0% { top: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

.stat-card {
  background: rgba(0, 0, 0, 0.3);
  padding: 20px 12px;
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.03);
  text-align: center;
  transition: all 0.3s ease;
  &:hover {
    background: rgba(0, 0, 0, 0.4);
    border-color: rgba(168, 85, 247, 0.2);
    transform: translateY(-5px);
  }
}

.orb-container {
  position: relative;
  width: 140px;
  height: 140px;
  margin: 0 auto;
  cursor: pointer;
  transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  
  &:hover {
    transform: scale(1.15);
  }
}

.main-orb {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #a855f7, #6366f1, #3b82f6);
  box-shadow: 0 0 40px rgba(168, 85, 247, 0.4);
}

.orb-glow {
  position: absolute;
  inset: -15px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
  animation: pulse-glow-orb 2s infinite;
}

@keyframes pulse-glow-orb {
  0%, 100% { transform: scale(1); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 0.8; }
}

.orb-icon {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 3.5rem;
  z-index: 5;
  filter: drop-shadow(0 0 10px rgba(0,0,0,0.5));
}

.btn-glow-primary {
  background: linear-gradient(135deg, #a855f7, #6366f1);
  box-shadow: 0 10px 25px -5px rgba(168, 85, 247, 0.4);
  transition: all 0.3s ease;
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px -5px rgba(168, 85, 247, 0.6);
  }
</style>

