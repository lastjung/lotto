<template>
  <div class="quantum-shuffle">
    <div class="quantum-beam" v-if="isAnimating"></div>
    <div class="quantum-balls">
      <div
        v-for="(ball, i) in displayNumbers"
        :key="i"
        class="quantum-ball"
        :class="[
          getBallColorClass(ball.number),
          { shuffling: ball.shuffling, locked: ball.locked }
        ]"
        :style="{ animationDelay: `${i * 0.1}s` }"
      >
        {{ ball.number }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onUnmounted } from 'vue'
import { useAnimation } from 'src/composables/useAnimation'

const props = defineProps({
  numbers: { type: Array, default: () => [] },
  isAnimating: { type: Boolean, default: false }
})

const emit = defineEmits(['complete'])

const { getBallColorClass, playPopSound, playCompleteSound, resumeAudio } = useAnimation()

const displayNumbers = ref([])
let shuffleInterval = null
let lockTimeouts = []

// Initialize with placeholder
function initDisplay(count = 6) {
  displayNumbers.value = Array(count).fill(0).map(() => ({
    number: Math.floor(Math.random() * 45) + 1,
    shuffling: false,
    locked: false
  }))
}

// Start shuffling animation
async function startShuffle(finalNumbers) {
  await resumeAudio()
  initDisplay(finalNumbers.length)
  
  // Start shuffling all balls
  displayNumbers.value.forEach(ball => {
    ball.shuffling = true
    ball.locked = false
  })
  
  // Random number cycling
  shuffleInterval = setInterval(() => {
    displayNumbers.value.forEach((ball, i) => {
      if (!ball.locked) {
        ball.number = Math.floor(Math.random() * 45) + 1
      }
    })
  }, 80)
  
  // Lock numbers one by one after delay
  const shuffleDuration = 1200
  const lockDelay = 200
  
  for (let i = 0; i < finalNumbers.length; i++) {
    await new Promise(resolve => {
      const timeout = setTimeout(() => {
        displayNumbers.value[i].shuffling = false
        displayNumbers.value[i].locked = true
        displayNumbers.value[i].number = finalNumbers[i]
        playPopSound()
        resolve()
      }, shuffleDuration + (i * lockDelay))
      lockTimeouts.push(timeout)
    })
  }
  
  clearInterval(shuffleInterval)
  playCompleteSound()
  emit('complete')
}

// Watch for numbers change - always trigger animation with sound
watch(() => props.numbers, (newNumbers, oldNumbers) => {
  if (newNumbers && newNumbers.length > 0) {
    // Check if numbers actually changed (not just initial load with same values)
    const numbersChanged = !oldNumbers || 
      oldNumbers.length !== newNumbers.length ||
      newNumbers.some((n, i) => n !== oldNumbers[i])
    
    if (numbersChanged) {
      // Always play animation with sound when new numbers arrive
      startShuffle(newNumbers)
    }
  }
}, { immediate: true })

// Cleanup
onUnmounted(() => {
  if (shuffleInterval) clearInterval(shuffleInterval)
  lockTimeouts.forEach(t => clearTimeout(t))
})

// Init
initDisplay()
</script>

<style lang="scss" scoped>
@import 'src/css/animation.scss';

.quantum-shuffle {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  position: relative;
}

.quantum-balls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.quantum-ball {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 22px;
  color: white;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  box-shadow:
    inset -4px -4px 10px rgba(0, 0, 0, 0.3),
    inset 4px 4px 10px rgba(255, 255, 255, 0.3),
    0 6px 15px rgba(0, 0, 0, 0.4);
  transition: all 0.3s ease;
}

.quantum-ball.shuffling {
  animation: quantum-flicker 0.08s linear infinite;
  opacity: 0.7;
}

@keyframes quantum-flicker {
  0%, 100% { opacity: 0.7; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.95); }
}

.quantum-ball.locked {
  animation: quantum-lock 0.4s ease-out forwards;
  opacity: 1;
}

@keyframes quantum-lock {
  0% { transform: scale(1.3); box-shadow: 0 0 30px rgba(255, 255, 255, 0.8); }
  100% { transform: scale(1); }
}

.quantum-beam {
  position: absolute;
  top: -20px;
  left: 50%;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, transparent, rgba(147, 51, 234, 0.8), transparent);
  transform: translateX(-50%);
  animation: beam-pulse 0.5s ease-in-out infinite;
}

@keyframes beam-pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}

@media (max-width: 640px) {
  .quantum-ball {
    width: 52px;
    height: 52px;
    font-size: 18px;
  }
}
</style>
