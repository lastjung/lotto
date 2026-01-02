<template>
  <div class="max-w-4xl mx-auto p-4 md:p-8">
      <h2 class="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <span>🤖</span> AI 분석 모델 가이드
      </h2>
      <p class="text-gray-400 mb-8">각 모델의 특징과 전략을 이해하고 자신에게 맞는 모델을 선택하세요.</p>

      <!-- AI Models Section -->
      <div class="glass-panel p-6 bg-[#1e293b]/40 border border-white/10 rounded-3xl mb-6">
          <h3 class="text-lg font-bold text-purple-400 mb-4">🧠 딥러닝 모델</h3>

          <div class="space-y-4">
              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-purple-500">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">⚡</span>
                      <span class="font-bold text-white">Transformer</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">Attention 메커니즘으로 과거 데이터의 "중요한 패턴"을 학습합니다.</p>
                  <p class="text-yellow-400 text-xs">💡 "모든 과거 데이터를 분석해서 중요한 패턴만 골라낸다"</p>
              </div>

              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-blue-500">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">🔄</span>
                      <span class="font-bold text-white">LSTM</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">시계열 순서를 기억하며 번호 흐름을 예측합니다.</p>
                  <p class="text-yellow-400 text-xs">💡 "최근 회차의 흐름을 따라가며 다음 번호를 예측"</p>
              </div>

              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-indigo-500">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">🌌</span>
                      <span class="font-bold text-white">AI Vector</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">번호들 사이의 숨겨진 관계를 고차원 공간(Vector)에서 입체적으로 분석합니다.</p>
                  <p class="text-indigo-400 text-xs">💡 "숫자 사이의 보이지 않는 관계를 입체적으로 분석"</p>
              </div>
          </div>
      </div>

      <!-- Statistical Models Section -->
      <div class="glass-panel p-6 bg-[#1e293b]/40 border border-white/10 rounded-3xl mb-6">
          <h3 class="text-lg font-bold text-green-400 mb-4">📊 통계 분석 모델</h3>

          <div class="space-y-4">
              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-orange-500">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">🔥</span>
                      <span class="font-bold text-white">Hot Trend</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">최근 30회차에서 자주 나온 "핫 번호"를 우선 선택합니다.</p>
                  <p class="text-orange-400 text-xs">💡 "요즘 자주 나오는 번호가 계속 나온다" (추세 추종)</p>
              </div>

              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-cyan-500">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">🧊</span>
                      <span class="font-bold text-white">Cold Theory</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">오랫동안 나오지 않은 "연체 번호"를 우선 선택합니다.</p>
                  <p class="text-cyan-400 text-xs">💡 "오래 안 나왔으니 나올 때가 됐다" (역발상)</p>
              </div>

              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-green-500">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">🎱</span>
                      <span class="font-bold text-white">Physics Bias</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">로또 기계의 물리적 편향(빈도, 위치)을 분석합니다.</p>
                  <p class="text-green-400 text-xs">💡 "자주 나온 번호가 계속 나온다" (기계적 편향)</p>
              </div>

              <div class="bg-white/5 rounded-2xl p-4 border-l-4 border-blue-400">
                  <div class="flex items-center gap-2 mb-2">
                      <span class="text-xl">⚖️</span>
                      <span class="font-bold text-white">Balanced Mix</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2">홀짝/고저/합계가 완벽하게 균형 잡힌 조합을 생성합니다.</p>
                  <p class="text-blue-400 text-xs">💡 "극단을 피하고 평균에 가까운 안정적인 조합"</p>
              </div>
          </div>
      </div>

      <!-- Comparison Table -->
      <div class="glass-panel p-6 bg-[#1e293b]/40 border border-white/10 rounded-3xl mb-6">
          <h3 class="text-lg font-bold text-yellow-400 mb-4">📋 모델 비교표</h3>

          <div class="overflow-x-auto">
              <table class="w-full text-sm">
                  <thead>
                      <tr class="border-b border-white/10">
                          <th class="text-left py-3 px-2 text-gray-400">모델</th>
                          <th class="text-left py-3 px-2 text-gray-400">전략</th>
                          <th class="text-left py-3 px-2 text-gray-400">추천 대상</th>
                      </tr>
                  </thead>
                  <tbody class="text-gray-300">
                      <tr class="border-b border-white/5">
                          <td class="py-3 px-2"><span class="text-purple-400">Transformer</span></td>
                          <td class="py-3 px-2">패턴 인식</td>
                          <td class="py-3 px-2">AI 추천을 원하는 분</td>
                      </tr>
                      <tr class="border-b border-white/5">
                          <td class="py-3 px-2"><span class="text-blue-400">LSTM</span></td>
                          <td class="py-3 px-2">흐름 예측</td>
                          <td class="py-3 px-2">최근 추세를 믿는 분</td>
                      </tr>
                      <tr class="border-b border-white/5">
                          <td class="py-3 px-2"><span class="text-indigo-400">AI Vector</span></td>
                          <td class="py-3 px-2">입체적 관계 분석</td>
                          <td class="py-3 px-2">데이터의 깊은 연결을 믿는 분</td>
                      </tr>
                      <tr class="border-b border-white/5">
                          <td class="py-3 px-2"><span class="text-orange-400">Hot Trend</span></td>
                          <td class="py-3 px-2">핫 번호</td>
                          <td class="py-3 px-2">"요즘 잘 나오는 번호" 선호</td>
                      </tr>
                      <tr class="border-b border-white/5">
                          <td class="py-3 px-2"><span class="text-cyan-400">Cold Theory</span></td>
                          <td class="py-3 px-2">연체 번호</td>
                          <td class="py-3 px-2">"이제 나올 때 됐다" 역발상</td>
                      </tr>
                      <tr class="border-b border-white/5">
                          <td class="py-3 px-2"><span class="text-green-400">Physics Bias</span></td>
                          <td class="py-3 px-2">빈도 편향</td>
                          <td class="py-3 px-2">기계적 패턴을 믿는 분</td>
                      </tr>
                      <tr>
                          <td class="py-3 px-2"><span class="text-blue-300">Balanced Mix</span></td>
                          <td class="py-3 px-2">완벽 균형</td>
                          <td class="py-3 px-2">안정적인 조합 선호</td>
                      </tr>
                  </tbody>
              </table>
          </div>
      </div>

      <!-- Tip Box -->
      <div class="bg-yellow-500/10 border border-yellow-500/30 rounded-2xl p-4">
          <div class="flex items-start gap-3">
              <span class="text-2xl">💡</span>
              <div>
                  <p class="text-yellow-400 font-bold mb-1">팁: 다양한 전략을 조합하세요!</p>
                  <p class="text-gray-300 text-sm">하나의 모델에만 의존하지 말고, 여러 모델의 결과를 비교해보세요.
                      예: Transformer 2게임 + Cold Theory 2게임 + Balanced Mix 1게임</p>
              </div>
          </div>
      </div>
  </div>
</template>

<script setup>
// Static Content
</script>

<style lang="scss" scoped>
.glass-panel {
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}
</style>
