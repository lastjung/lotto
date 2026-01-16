<template>
  <div class="slot-machine-wrapper" :class="{ 'slot-complete': isComplete }">
    <div class="slot-machine">
      <div class="slot-header">
        <span class="slot-title">🎰 LUCKY DRAW 🎰</span>
      </div>
      <div class="slot-reels">
        <div v-for="(reel, i) in reels" :key="i" class="slot-reel">
          <div class="reel-container">
            <div 
              class="reel-strip" 
              :class="{ spinning: reel.spinning, stopped: reel.stopped }"
              :id="`reel-${i}`"
            >
              <div
                v-for="(num, j) in reel.numbers"
                :key="j"
                class="reel-number"
                :class="getBallColorClass(num)"
              >
                <span>{{ num }}</span>
              </div>
            </div>
          </div>
          <div class="reel-frame"></div>
        </div>
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

const { getBallColorClass, playSpinSound, playCompleteSound, resumeAudio } = useAnimation()

const reels = ref([])
const isComplete = ref(false)
let animationTimeouts = []

// Create reel numbers (random + final)
function createReelNumbers(finalNum) {
  const randomNums = Array(20).fill(0).map(() => Math.floor(Math.random() * 45) + 1)
  randomNums.push(finalNum)
  return randomNums
}

// Initialize reels
function initReels(numbers) {
  reels.value = numbers.map(num => ({
    numbers: createReelNumbers(num),
    spinning: false,
    stopped: false,
    final: num
  }))
}

// Run animation
async function animate(numbers) {
  await resumeAudio()
  isComplete.value = false
  initReels(numbers)
  
  // Start spinning all
  playSpinSound()
  reels.value.forEach(reel => {
    reel.spinning = true
    reel.stopped = false
  })
  
  // Stop reels one by one
  const spinDuration = 2000
  const reelDelay = 300
  
  for (let i = 0; i < reels.value.length; i++) {
    await new Promise(resolve => {
      const timeout = setTimeout(() => {
        reels.value[i].spinning = false
        reels.value[i].stopped = true
        resolve()
      }, spinDuration + (i * reelDelay))
      animationTimeouts.push(timeout)
    })
  }
  
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
  animationTimeouts.forEach(t => clearTimeout(t))
})

// Init with placeholder
initReels([1, 2, 3, 4, 5, 6])
</script>

<style lang="scss" scoped>
@import 'src/css/animation.scss';

.slot-machine-wrapper {
  display: flex;
  justify-content: center;
  padding: 1rem;
}
</style>
