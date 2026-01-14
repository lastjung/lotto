<template>
  <div class="lotto-machine">
    <div class="machine-dome">
      <div class="mixing-balls" :class="{ mixing: isAnimating }">
        <div
          v-for="(ball, i) in mixingBalls"
          :key="i"
          class="mixing-ball"
          :class="ball.colorClass"
          :style="{ '--delay': `${i * 0.1}s`, '--x': `${ball.x}%`, '--y': `${ball.y}%` }"
        ></div>
      </div>
    </div>
    <div class="machine-chute">
      <div class="chute-opening"></div>
    </div>
    <div class="revealed-balls">
      <div
        v-for="(num, i) in revealedNumbers"
        :key="i"
        class="revealed-ball pop-in"
        :class="getBallColorClass(num)"
        :style="{ animationDelay: `${i * 0.4}s` }"
      >
        {{ num }}
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

const mixingBalls = ref([])
const revealedNumbers = ref([])
let revealTimeout = null

// Generate random mixing balls
function generateMixingBalls() {
  const colors = ['ball-yellow', 'ball-blue', 'ball-red', 'ball-gray', 'ball-green']
  mixingBalls.value = Array(12).fill(0).map((_, i) => ({
    colorClass: colors[i % 5],
    x: Math.random() * 100,
    y: Math.random() * 100
  }))
}

// Reveal numbers one by one
async function revealNumbers(numbers) {
  await resumeAudio()
  revealedNumbers.value = []
  
  for (let i = 0; i < numbers.length; i++) {
    await new Promise(resolve => {
      revealTimeout = setTimeout(() => {
        revealedNumbers.value.push(numbers[i])
        playPopSound()
        resolve()
      }, 400)
    })
  }
  
  playCompleteSound()
  emit('complete')
}

// Watch for animation start
watch(() => props.isAnimating, (val) => {
  if (val) {
    generateMixingBalls()
    revealedNumbers.value = []
  }
})

// Watch for numbers to reveal
watch(() => props.numbers, (newNumbers) => {
  if (newNumbers && newNumbers.length > 0 && !props.isAnimating) {
    revealNumbers(newNumbers)
  }
}, { immediate: true })

onUnmounted(() => {
  if (revealTimeout) clearTimeout(revealTimeout)
})

// Init
generateMixingBalls()
</script>

<style lang="scss" scoped>
@import 'src/css/animation.scss';
</style>
