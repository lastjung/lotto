<template>
  <div 
    class="lotto-ball-v2"
    :class="colorClass"
    :style="ballStyle"
  >
    <div class="ball-inner">
      {{ number }}
    </div>
    <div class="ball-highlight"></div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  number: {
    type: [Number, String],
    required: true
  },
  size: {
    type: String,
    default: '40px'
  }
})

const colorClass = computed(() => {
  const num = parseInt(props.number)
  if (isNaN(num)) return 'bg-gray'
  if (num <= 10) return 'ball-yellow'
  if (num <= 20) return 'ball-blue'
  if (num <= 30) return 'ball-red'
  if (num <= 40) return 'ball-gray'
  return 'ball-green'
})

const ballStyle = computed(() => ({
  width: props.size,
  height: props.size,
  fontSize: `calc(${props.size} * 0.45)`
}))
</script>

<style lang="scss" scoped>
.lotto-ball-v2 {
  position: relative;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  color: white;
  text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  box-shadow: 
    inset 0 -4px 6px rgba(0,0,0,0.3),
    inset 0 4px 6px rgba(255,255,255,0.2),
    0 4px 10px rgba(0,0,0,0.4);
  transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  cursor: default;
  user-select: none;

  &:hover {
    transform: scale(1.1) translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.5);
    z-index: 10;
  }
}

.ball-inner {
  z-index: 2;
  position: relative;
}

.ball-highlight {
  position: absolute;
  top: 15%;
  left: 15%;
  width: 30%;
  height: 30%;
  background: rgba(255,255,255,0.4);
  filter: blur(2px);
  border-radius: 50%;
  pointer-events: none;
}

.ball-yellow {
  background: radial-gradient(circle at 30% 30%, #fde047, #ca8a04);
  color: #422006;
  text-shadow: none;
}

.ball-blue {
  background: radial-gradient(circle at 30% 30%, #60a5fa, #1d4ed8);
}

.ball-red {
  background: radial-gradient(circle at 30% 30%, #f87171, #b91c1c);
}

.ball-gray {
  background: radial-gradient(circle at 30% 30%, #9ca3af, #374151);
}

.ball-green {
  background: radial-gradient(circle at 30% 30%, #4ade80, #15803d);
}

.bg-gray {
  background: #333;
}
</style>
