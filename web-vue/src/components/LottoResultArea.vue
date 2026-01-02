<template>
  <div class="lotto-result-area glass-panel p-8 relative overflow-hidden">
    <!-- Scanner Glow Effect -->
    <div v-if="scanning" class="scanner-line"></div>
    
    <div class="flex flex-col items-center justify-center min-h-[400px]">
      <div v-if="!results && !generating" class="text-center">
        <div class="orb-container mb-8" @click="$emit('generate')">
          <div class="main-orb"></div>
          <div class="orb-glow"></div>
          <span class="orb-icon">🔮</span>
        </div>
        <h2 class="text-2xl font-bold text-white mb-2">Ready for Analysis?</h2>
        <p class="text-gray-400 max-w-xs mx-auto mb-8">Select your AI model and initiate the quantum generation process.</p>
        <q-btn
          unelevated
          rounded
          padding="md xl"
          class="btn-glow-primary text-white font-bold"
          label="GENERATE LUCKY NUMBERS"
          :loading="generating"
          @click="$emit('generate')"
        />
      </div>

      <div v-else class="w-full">
        <!-- Result Header -->
        <div class="flex justify-between items-center mb-8 border-b border-white/5 pb-4">
          <div class="flex items-center gap-3">
             <div class="w-2 h-2 rounded-full bg-pink-500 shadow-glow-pink"></div>
             <h3 class="font-bold text-white tracking-wider">GENERATION COMPLETE</h3>
          </div>
          <div class="flex gap-2">
            <span class="text-[10px] bg-blue-500/20 text-blue-400 px-2 py-1 rounded border border-blue-500/30 font-mono">
              MODEL: {{ modelName }}
            </span>
          </div>
        </div>

        <!-- Balls Container -->
        <div class="flex justify-center gap-4 mb-12 flex-wrap">
          <LottoBall 
            v-for="(num, i) in results" 
            :key="i"
            :number="num"
            size="64px"
            class="pop-animation"
            :style="{ animationDelay: `${i * 0.1}s` }"
          />
        </div>

        <!-- Analysis Grid -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
           <div class="stat-card">
              <span class="text-[10px] text-gray-500 uppercase">Sum Total</span>
              <div class="text-lg font-bold text-white">{{ analysis?.sum || '-' }}</div>
           </div>
           <div class="stat-card">
              <span class="text-[10px] text-gray-500 uppercase">AC Value</span>
              <div class="text-lg font-bold text-white">{{ analysis?.ac_value || '-' }}</div>
           </div>
           <div class="stat-card">
              <span class="text-[10px] text-gray-500 uppercase">Odd:Even</span>
              <div class="text-lg font-bold text-white">{{ analysis?.odd_even || '-' }}</div>
           </div>
           <div class="stat-card">
              <span class="text-[10px] text-gray-400 uppercase">Confidence</span>
              <div class="text-lg font-bold text-green-400">98.4%</div>
           </div>
        </div>

        <div class="mt-10 text-center">
           <q-btn
              flat
              rounded
              color="white"
              icon="refresh"
              label="Generate Again"
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
  background: rgba(30, 41, 59, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 32px;
}

.scanner-line {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, transparent, #3b82f6, transparent);
  box-shadow: 0 0 15px #3b82f6;
  z-index: 20;
  animation: scan 2s linear infinite;
}

@keyframes scan {
  0% { top: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

.stat-card {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.03);
  text-align: center;
}

.orb-container {
  position: relative;
  width: 120px;
  height: 120px;
  margin: 0 auto;
  cursor: pointer;
  transition: transform 0.3s ease;
  
  &:hover {
    transform: scale(1.1);
  }
}

.main-orb {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #a855f7, #6366f1);
  box-shadow: 0 0 30px rgba(168, 85, 247, 0.5);
  animation: floating 3s ease-in-out infinite;
}

.orb-glow {
  position: absolute;
  top: -10px;
  left: -10px;
  right: -10px;
  bottom: -10px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.2) 0%, transparent 70%);
  animation: pulse-glow 2s infinite;
}

.orb-icon {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 3rem;
  z-index: 5;
}

.shadow-glow-pink {
  box-shadow: 0 0 10px #ec4899;
}

.pop-animation {
  animation: pop 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) backwards;
}

@keyframes pop {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
</style>
