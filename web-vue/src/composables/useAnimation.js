/**
 * useAnimation - Animation Manager Composable
 * Ported from legacy web/js/animation.js AnimationManager
 */
import { ref, computed, onMounted } from 'vue'

const STORAGE_KEYS = {
  TYPE: 'l_animation_type',
  SOUND: 'l_sound_enabled'
}

// Singleton state (shared across all components)
const currentType = ref('quantum_shuffle')
const soundEnabled = ref(true)
let audioCtx = null

export function useAnimation() {
  const availableTypes = [
    { id: 'lottery_ball', name: '🎱 로또볼 추첨기', desc: '공이 튀며 나오는 실제 추첨 효과' },
    { id: 'slot_machine', name: '🎰 슬롯머신', desc: '카지노 스타일 릴 회전' },
    { id: 'ai_scanner', name: '🔬 AI 스캐너', desc: '미래형 스캔 & 락인 효과' },
    { id: 'quantum_shuffle', name: '🔮 퀀텀 셔플', desc: '양자 빔과 함께 숫자가 회전' }
  ]

  const currentTypeName = computed(() => {
    const found = availableTypes.find(t => t.id === currentType.value)
    return found ? found.name : 'Unknown'
  })

  // Initialize from localStorage
  onMounted(() => {
    const savedType = localStorage.getItem(STORAGE_KEYS.TYPE)
    if (savedType && availableTypes.some(t => t.id === savedType)) {
      currentType.value = savedType
    }

    const savedSound = localStorage.getItem(STORAGE_KEYS.SOUND)
    if (savedSound !== null) {
      soundEnabled.value = savedSound === 'true'
    }
  })

  function setType(type) {
    if (availableTypes.some(t => t.id === type)) {
      currentType.value = type
      localStorage.setItem(STORAGE_KEYS.TYPE, type)
    }
  }

  function toggleSound(enabled) {
    soundEnabled.value = enabled
    localStorage.setItem(STORAGE_KEYS.SOUND, enabled.toString())
  }

  // Get or create AudioContext
  function getAudioContext() {
    if (!audioCtx) {
      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)()
      } catch (e) {
        console.warn('Web Audio API not supported')
        return null
      }
    }
    return audioCtx
  }

  // Resume AudioContext (needed for user gesture requirement)
  async function resumeAudio() {
    const ctx = getAudioContext()
    if (ctx && ctx.state === 'suspended') {
      await ctx.resume()
    }
  }

  // Play pop sound
  function playPopSound() {
    if (!soundEnabled.value) return
    const ctx = getAudioContext()
    if (!ctx) return

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.connect(gain)
    gain.connect(ctx.destination)

    osc.frequency.setValueAtTime(800, ctx.currentTime)
    osc.frequency.exponentialRampToValueAtTime(300, ctx.currentTime + 0.1)

    gain.gain.setValueAtTime(0.3, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1)

    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + 0.1)
  }

  // Play complete celebration sound
  function playCompleteSound() {
    if (!soundEnabled.value) return
    const ctx = getAudioContext()
    if (!ctx) return

    const notes = [523, 659, 784, 1047] // C5, E5, G5, C6

    notes.forEach((freq, i) => {
      setTimeout(() => {
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()

        osc.connect(gain)
        gain.connect(ctx.destination)

        osc.frequency.setValueAtTime(freq, ctx.currentTime)
        osc.type = 'sine'

        gain.gain.setValueAtTime(0.2, ctx.currentTime)
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3)

        osc.start(ctx.currentTime)
        osc.stop(ctx.currentTime + 0.3)
      }, i * 100)
    })
  }

  // Play scan/lock sound for AI Scanner
  function playScanSound() {
    if (!soundEnabled.value) return
    const ctx = getAudioContext()
    if (!ctx) return

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.frequency.setValueAtTime(1200, ctx.currentTime)
    osc.type = 'sine'
    gain.gain.setValueAtTime(0.1, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1)
    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + 0.1)
  }

  function playLockSound() {
    if (!soundEnabled.value) return
    const ctx = getAudioContext()
    if (!ctx) return

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.frequency.setValueAtTime(880, ctx.currentTime)
    osc.frequency.setValueAtTime(1320, ctx.currentTime + 0.05)
    osc.type = 'square'
    gain.gain.setValueAtTime(0.2, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15)
    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + 0.15)
  }

  // Play slot machine sounds
  function playSpinSound() {
    if (!soundEnabled.value) return
    const ctx = getAudioContext()
    if (!ctx) return

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.connect(gain)
    gain.connect(ctx.destination)

    osc.frequency.setValueAtTime(150, ctx.currentTime)
    osc.type = 'sawtooth'

    gain.gain.setValueAtTime(0.1, ctx.currentTime)
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 2)

    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + 2)
  }

  // Get ball color class based on number
  function getBallColorClass(num) {
    if (num <= 10) return 'ball-yellow'
    if (num <= 20) return 'ball-blue'
    if (num <= 30) return 'ball-red'
    if (num <= 40) return 'ball-gray'
    return 'ball-green'
  }

  return {
    currentType,
    currentTypeName,
    soundEnabled,
    availableTypes,
    setType,
    toggleSound,
    resumeAudio,
    playPopSound,
    playCompleteSound,
    playScanSound,
    playLockSound,
    playSpinSound,
    getBallColorClass
  }
}
