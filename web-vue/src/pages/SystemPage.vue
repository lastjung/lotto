<template>
  <div class="max-w-4xl mx-auto p-4 md:p-8">
      <div class="flex items-center justify-between mb-2">
          <h2 class="text-2xl font-bold text-white flex items-center gap-3">
              <span>⚙️</span> System Configuration
          </h2>
          <button @click="$router.push('/')"
              class="p-2 rounded-full hover:bg-white/10 transition-colors" title="대시보드로 이동">
              <svg class="w-6 h-6 text-gray-400 hover:text-white" fill="none" stroke="currentColor"
                  viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M6 18L18 6M6 6l12 12" />
              </svg>
          </button>
      </div>
      <p class="text-gray-500 mb-8 text-sm">AI Analysis Engine and Animation Settings</p>

      <!-- Animation Selector -->
      <div class="glass-panel p-6 bg-[#1e293b]/40 border border-white/10 rounded-3xl mb-6">
          <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <span>🎬</span> Animation Style
          </h3>
          <p class="text-gray-500 text-sm mb-4">번호 생성 시 표시되는 애니메이션을 선택하세요</p>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <!-- Lottery Ball -->
              <div 
                class="animation-option cursor-pointer p-4 rounded-xl border-2 transition-all hover:scale-105"
                :class="selectedAnimation === 'lottery_ball' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-600 bg-gray-800/50 hover:border-purple-400'"
                @click="selectedAnimation = 'lottery_ball'"
              >
                  <div class="flex items-center gap-3 mb-2">
                      <span class="text-3xl">🎱</span>
                      <div>
                          <h4 class="font-bold text-white">로또볼 추첨기</h4>
                          <p class="text-xs text-gray-400">공이 튀며 나오는 실제 추첨 효과</p>
                      </div>
                  </div>
                  <div v-if="selectedAnimation === 'lottery_ball'" class="selected-badge text-xs text-purple-400 font-medium">✓ 선택됨</div>
              </div>

              <!-- Slot Machine -->
              <div 
                class="animation-option cursor-pointer p-4 rounded-xl border-2 transition-all hover:scale-105"
                :class="selectedAnimation === 'slot_machine' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-600 bg-gray-800/50 hover:border-purple-400'"
                @click="selectedAnimation = 'slot_machine'"
              >
                  <div class="flex items-center gap-3 mb-2">
                      <span class="text-3xl">🎰</span>
                      <div>
                          <h4 class="font-bold text-white">슬롯머신</h4>
                          <p class="text-xs text-gray-400">카지노 스타일 릴 회전</p>
                      </div>
                  </div>
                  <div v-if="selectedAnimation === 'slot_machine'" class="selected-badge text-xs text-purple-400 font-medium">✓ 선택됨</div>
              </div>

              <!-- AI Scanner -->
              <div 
                class="animation-option cursor-pointer p-4 rounded-xl border-2 transition-all hover:scale-105"
                :class="selectedAnimation === 'ai_scanner' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-600 bg-gray-800/50 hover:border-purple-400'"
                @click="selectedAnimation = 'ai_scanner'"
              >
                  <div class="flex items-center gap-3 mb-2">
                      <span class="text-3xl">🔬</span>
                      <div>
                          <h4 class="font-bold text-white">AI 스캐너</h4>
                          <p class="text-xs text-gray-400">미래형 스캔 & 락인 효과</p>
                      </div>
                  </div>
                  <div v-if="selectedAnimation === 'ai_scanner'" class="selected-badge text-xs text-purple-400 font-medium">✓ 선택됨</div>
              </div>

              <!-- Quantum Shuffle (NEW) -->
              <div 
                class="animation-option cursor-pointer p-4 rounded-xl border-2 transition-all hover:scale-105"
                :class="selectedAnimation === 'quantum_shuffle' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-600 bg-gray-800/50 hover:border-purple-400'"
                @click="selectedAnimation = 'quantum_shuffle'"
              >
                  <div class="flex items-center gap-3 mb-2">
                      <span class="text-3xl">🔮</span>
                      <div>
                          <h4 class="font-bold text-white">퀀텀 셔플</h4>
                          <p class="text-xs text-gray-400">양자 빔과 함께 숫자가 회전</p>
                      </div>
                  </div>
                  <div v-if="selectedAnimation === 'quantum_shuffle'" class="selected-badge text-xs text-purple-400 font-medium">✓ 선택됨</div>
              </div>
          </div>
      </div>

      <!-- Sound Toggle -->
      <div class="glass-panel p-6 bg-[#1e293b]/40 border border-white/10 rounded-3xl">
          <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <span>🔊</span> Sound Effects
          </h3>
          <div class="flex items-center justify-between">
              <div>
                  <p class="text-white">애니메이션 효과음</p>
                  <p class="text-gray-500 text-sm">번호 생성 시 사운드 효과 재생</p>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" v-model="soundEnabled" class="sr-only peer">
                  <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-500">
                  </div>
              </label>
          </div>
      </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'

const selectedAnimation = ref('lottery_ball')
const soundEnabled = ref(true)

// Persist settings
onMounted(() => {
  const savedAnim = localStorage.getItem('l_animation_type')
  if (savedAnim) selectedAnimation.value = savedAnim
  
  const savedSound = localStorage.getItem('l_sound_enabled')
  if (savedSound !== null) soundEnabled.value = savedSound === 'true'
})

watch(selectedAnimation, (val) => {
  localStorage.setItem('l_animation_type', val)
  // dispatch event for global listener if needed, but localStorage is usually enough
})

watch(soundEnabled, (val) => {
  localStorage.setItem('l_sound_enabled', val)
})
</script>

<style lang="scss" scoped>
.glass-panel {
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}
</style>
