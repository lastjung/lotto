<template>
  <div class="ai-scanner-wrapper" :class="{ 'scanner-complete': isComplete }">
    <div class="ai-scanner">
      <div class="scanner-header">
        <span class="scanner-title">🔬 AI ANALYSIS</span>
        <span class="scanner-status">{{ statusText }}</span>
      </div>
      <div class="scanner-slots">
        <div
          v-for="(slot, i) in slots"
          :key="i"
          class="scanner-slot"
          :class="{ locked: slot.locked }"
        >
          <div class="scanner-display" :class="slot.locked ? getBallColorClass(slot.final) : ''">
            <span class="scanning-number">{{ slot.display }}</span>
          </div>
          <div class="scanner-glow"></div>
        </div>
      </div>
      <div class="scanner-progress">
        <div class="progress-bar" :style="{ width: `${progress}%` }"></div>
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

const { getBallColorClass, playScanSound, playLockSound, playCompleteSound, resumeAudio } = useAnimation()

const slots = ref([])
const progress = ref(0)
const statusText = ref('READY')
const isComplete = ref(false)
let scanInterval = null
let animationTimeouts = []

// Initialize slots
function initSlots(numbers) {
  slots.value = numbers.map(num => ({
    final: num,
    display: '--',
    locked: false
  }))
}

// Run animation
async function animate(numbers) {
  await resumeAudio()
  isComplete.value = false
  progress.value = 0
  statusText.value = 'SCANNING...'
  initSlots(numbers)
  
  // Scanning phase
  scanInterval = setInterval(() => {
    slots.value.forEach(slot => {
      if (!slot.locked) {
        slot.display = (Math.floor(Math.random() * 45) + 1).toString().padStart(2, '0')
      }
    })
    playScanSound()
  }, 80)
  
  // Progress animation
  const scanDuration = 1500
  const startTime = Date.now()
  const progressInterval = setInterval(() => {
    const elapsed = Date.now() - startTime
    progress.value = Math.min((elapsed / scanDuration) * 100, 100)
    if (elapsed >= scanDuration) {
      clearInterval(progressInterval)
    }
  }, 50)
  
  // Wait for scan
  await new Promise(resolve => {
    const timeout = setTimeout(resolve, scanDuration)
    animationTimeouts.push(timeout)
  })
  
  clearInterval(scanInterval)
  statusText.value = 'LOCKING...'
  
  // Lock phase
  for (let i = 0; i < slots.value.length; i++) {
    await new Promise(resolve => {
      const timeout = setTimeout(() => {
        slots.value[i].locked = true
        slots.value[i].display = slots.value[i].final.toString().padStart(2, '0')
        playLockSound()
        resolve()
      }, 300)
      animationTimeouts.push(timeout)
    })
  }
  
  statusText.value = 'COMPLETE!'
  isComplete.value = true
  playCompleteSound()
  emit('complete')
}

// Watch for numbers change - trigger animation with sound
watch(() => props.numbers, (newNumbers, oldNumbers) => {
  if (newNumbers && newNumbers.length > 0) {
    const numbersChanged = !oldNumbers || 
      oldNumbers.length !== newNumbers.length ||
      newNumbers.some((n, i) => n !== oldNumbers[i])
    
    if (numbersChanged) {
      animate(newNumbers)
    }
  }
}, { immediate: true })

// Cleanup
onUnmounted(() => {
  if (scanInterval) clearInterval(scanInterval)
  animationTimeouts.forEach(t => clearTimeout(t))
})

// Init with placeholder
initSlots([0, 0, 0, 0, 0, 0])
</script>

<style lang="scss" scoped>
@import 'src/css/animation.scss';

.ai-scanner-wrapper {
  display: flex;
  justify-content: center;
  padding: 1rem;
}
</style>
