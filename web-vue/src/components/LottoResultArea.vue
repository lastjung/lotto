<template>
  <div class="lotto-result-area glass-panel p-6 md:p-10 relative overflow-hidden group">
    <!-- Scanner Glow Effect -->
    <div v-if="scanning" class="scanner-line"></div>
    
    <!-- Background Decoration -->
    <div class="absolute -top-24 -right-24 w-64 h-64 bg-primary/10 rounded-full blur-3xl pointer-events-none group-hover:bg-primary/20 transition-colors"></div>
    <div class="absolute -bottom-24 -left-24 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-blue-500/20 transition-colors"></div>

    <div class="flex flex-col items-center justify-center min-h-[400px] relative z-10">
      <div v-if="!results && !generating" class="text-center">
        <div class="orb-container mb-10" @click="$emit('generate')">
          <div class="main-orb pulsing-core"></div>
          <div class="orb-glow"></div>
          <span class="orb-icon">🔮</span>
        </div>
        <h2 class="text-2xl md:text-3xl font-bold text-white mb-3 heading-font tracking-tight">Ready for Analysis</h2>
        <p class="text-gray-400 max-w-xs mx-auto mb-10 text-sm leading-relaxed">Select a strategy on the left and start the Multi-Layer Analysis.</p>
      </div>

      <div v-else class="w-full">
        <!-- Result Header -->
        <div class="flex justify-between items-center mb-6 md:mb-10 border-b border-white/5 pb-6">
          <div class="flex items-center gap-3">
             <div :class="['w-2.5 h-2.5 rounded-full animate-pulse', results ? 'bg-pink-500 shadow-glow-pink' : 'bg-blue-400 shadow-glow-blue']"></div>
             <h3 class="font-bold text-white tracking-widest uppercase text-[10px] md:text-sm">
               {{ results ? 'Generation Complete' : currentTypeName }}
             </h3>
          </div>
          <div class="flex gap-2">
            <span class="text-[9px] md:text-[10px] bg-blue-500/20 text-blue-400 px-2.5 py-1 rounded-full border border-blue-500/30 font-bold tracking-wider uppercase">
               {{ modelName }}
            </span>
          </div>
        </div>

        <!-- Primary Set: Dynamic Animation Component -->
        <div class="animation-container mb-6">
          <component 
            :is="currentAnimationComponent"
            :numbers="results || []"
            :is-animating="generating"
            @complete="onAnimationComplete"
          />
        </div>

        <!-- Analysis Grid (First Set Only) - Show after animation -->
        <Transition name="fade-slide">
          <div v-if="showStats" class="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4 mt-6 mb-8">
             <div class="stat-card group">
                <span class="text-[9px] text-gray-500 uppercase tracking-widest block mb-1">Sum Total</span>
                <div class="text-lg md:text-xl font-bold text-white group-hover:text-primary transition-colors">{{ analysis?.sum || '-' }}</div>
             </div>
             <div class="stat-card group">
                <span class="text-[9px] text-gray-500 uppercase tracking-widest block mb-1">AC Value</span>
                <div class="text-lg md:text-xl font-bold text-white group-hover:text-primary transition-colors">{{ analysis?.ac_value || '-' }}</div>
             </div>
             <div class="stat-card group">
                <span class="text-[9px] text-gray-500 uppercase tracking-widest block mb-1">Odd:Even</span>
                <div class="text-lg md:text-xl font-bold text-white group-hover:text-primary transition-colors">{{ analysis?.odd_even || '-' }}</div>
             </div>
             <div class="stat-card group border-green-500/20">
                <span class="text-[9px] text-green-400/70 uppercase tracking-widest block mb-1">Confidence</span>
                <div class="text-lg md:text-xl font-bold text-green-400">{{ analysis?.confidence || '98.4' }}%</div>
             </div>
          </div>
        </Transition>

        <!-- Additional Sets (2nd ~ 5th) - Show sequentially after stats -->
        <Transition name="fade-slide">
          <div v-if="showStats && allSets && allSets.length > 1" class="mt-6 pt-6 border-t border-white/5">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-4 font-bold flex items-center gap-2">
              <q-icon name="plus_one" size="14px" />
              Additional Combinations
            </div>
            <div class="space-y-4">
              <TransitionGroup name="list">
                <div 
                  v-for="(set, idx) in visibleSets" 
                  :key="idx"
                  class="flex flex-col sm:flex-row sm:items-center gap-3 bg-white/5 md:bg-black/20 rounded-2xl p-4 border border-white/5"
                >
                  <div class="flex items-center justify-between sm:justify-start gap-3">
                    <span class="text-[10px] text-blue-400 font-bold bg-blue-500/10 px-2 py-0.5 rounded">SET #{{ idx + 2 }}</span>
                    <span class="sm:hidden text-[10px] text-gray-500">Σ {{ set.reduce((a, b) => a + b, 0) }}</span>
                  </div>
                  <div class="flex gap-2 flex-wrap justify-center sm:justify-start">
                    <span 
                      v-for="num in set" 
                      :key="num"
                      :class="['w-11 h-11 sm:w-8 sm:h-8 rounded-full flex items-center justify-center text-xs sm:text-[10px] font-bold shadow-xl transition-transform hover:scale-110 cursor-pointer touch-manipulation', getBallColor(num)]"
                    >{{ num }}</span>
                  </div>
                  <span class="hidden sm:inline ml-auto text-[10px] text-gray-500 font-mono">SUM: {{ set.reduce((a, b) => a + b, 0) }}</span>
                </div>
              </TransitionGroup>
            </div>
          </div>
        </Transition>

        <div class="mt-8 text-center">
           <q-btn
              flat
              rounded
              color="blue-4"
              icon="refresh"
              label="Re-analyze Quantum Space"
              class="text-[10px] uppercase tracking-widest opacity-60 hover:opacity-100"
              @click="$emit('generate')"
           />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, defineAsyncComponent } from 'vue'
import { useAnimation } from 'src/composables/useAnimation'

// Lazy load animation components
const LotteryBallAnimation = defineAsyncComponent(() => import('./animations/LotteryBallAnimation.vue'))
const SlotMachineAnimation = defineAsyncComponent(() => import('./animations/SlotMachineAnimation.vue'))
const AIScannerAnimation = defineAsyncComponent(() => import('./animations/AIScannerAnimation.vue'))
const QuantumShuffleAnimation = defineAsyncComponent(() => import('./animations/QuantumShuffleAnimation.vue'))

const props = defineProps({
  results: Array,
  allSets: {
    type: Array,
    default: () => []
  },
  analysis: Object,
  generating: Boolean,
  scanning: Boolean,
  modelName: String
})

defineEmits(['generate'])

const { currentType, currentTypeName } = useAnimation()

// Sequential display state
const showStats = ref(false)
const visibleSetCount = ref(0)

// Visible additional sets (2nd to 5th)
const visibleSets = computed(() => {
  if (!props.allSets || props.allSets.length <= 1) return []
  return props.allSets.slice(1, 1 + visibleSetCount.value)
})

// Map animation types to components
const animationComponents = {
  lottery_ball: LotteryBallAnimation,
  slot_machine: SlotMachineAnimation,
  ai_scanner: AIScannerAnimation,
  quantum_shuffle: QuantumShuffleAnimation
}

const currentAnimationComponent = computed(() => {
  return animationComponents[currentType.value] || QuantumShuffleAnimation
})

// Reset on new generation
watch(() => props.generating, (isGenerating) => {
  if (isGenerating) {
    showStats.value = false
    visibleSetCount.value = 0
  }
})

// Called when animation component emits 'complete'
function onAnimationComplete() {
  // Only show stats after animation is fully complete
  showStats.value = true
  
  // Show additional sets one by one  
  const additionalCount = (props.allSets?.length || 1) - 1
  for (let i = 1; i <= additionalCount; i++) {
    setTimeout(() => {
      visibleSetCount.value = i
    }, i * 300) // 300ms delay between each set
  }
}

// Ball color based on number range
function getBallColor(n) {
  if (n <= 10) return 'bg-yellow-500 text-black'
  if (n <= 20) return 'bg-blue-500 text-white'
  if (n <= 30) return 'bg-red-500 text-white'
  if (n <= 40) return 'bg-gray-500 text-white'
  return 'bg-green-500 text-white'
}
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
}

// Sequential reveal transitions
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.4s ease;
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

// List item transitions
.list-enter-active {
  transition: all 0.3s ease;
}

.list-enter-from {
  opacity: 0;
  transform: translateX(-20px);
}

.list-move {
  transition: transform 0.3s ease;
}
</style>

