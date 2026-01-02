<template>
  <q-layout view="lHh Lpr lFf" class="bg-dark-page text-white">
    <!-- Desktop Sidebar (Drawer) -->
    <q-drawer
      v-model="leftDrawerOpen"
      show-if-above
      :width="280"
      class="bg-dark shadow-24"
      style="border-right: 1px solid rgba(255, 255, 255, 0.05)"
    >
      <div class="column h-full">
        <!-- Logo Section -->
        <div class="q-pa-lg">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg transform hover:rotate-12 transition-transform">
              <span class="text-white text-xl">🧬</span>
            </div>
            <div>
              <div class="text-lg font-bold tracking-tight text-white heading-font">LottoQuantAI</div>
              <div class="text-[10px] text-blue-400 font-bold tracking-widest uppercase">Deep Space V2</div>
            </div>
          </div>
        </div>

        <!-- Lottery Selector -->
        <div class="px-6 mb-6">
          <q-select
            filled
            v-model="currentLottery"
            :options="lotteryOptions"
            option-label="name"
            option-value="id"
            dark
            dense
            rounded
            standout
            class="lottery-select"
            bg-color="void-navy"
            behavior="menu"
            @update:model-value="onLotteryChange"
          >
            <template v-slot:prepend>
              <q-icon :name="currentLottery?.icon || 'public'" color="blue-4" size="xs" />
            </template>
          </q-select>
        </div>

        <!-- Navigation -->
        <q-list class="flex-grow px-4 space-y-2">
          <q-item
            v-for="item in menuItems"
            :key="item.id"
            clickable
            v-ripple
            :to="item.to"
            active-class="nav-active"
            class="rounded-xl text-grey-5 nav-item"
          >
            <q-item-section avatar>
              <span class="text-xl">{{ item.icon }}</span>
            </q-item-section>
            <q-item-section>
              <q-item-label class="font-medium">{{ item.label }}</q-item-label>
            </q-item-section>
          </q-item>
        </q-list>

        <!-- System Status -->
        <div class="p-6 border-t border-white/5 mt-auto">
          <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
            <div class="flex items-center gap-2 mb-2">
              <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span class="text-xs font-bold text-green-400 uppercase tracking-tighter">System Online</span>
            </div>
            <div class="text-[10px] text-gray-500 flex flex-col gap-1">
              <div class="flex justify-between">
                <span>Engine:</span>
                <span class="text-gray-300">v2.1.0 (ONNX)</span>
              </div>
              <div class="flex justify-between">
                <span>Agent:</span>
                <span class="text-gray-300">Connected</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </q-drawer>

    <!-- Page Content -->
    <q-page-container>
      <!-- Mobile Header -->
      <q-header v-if="$q.screen.lt.md" class="bg-dark-page/80 backdrop-blur-md border-b border-white/5">
        <q-toolbar>
          <q-btn
            flat
            dense
            round
            icon="menu"
            aria-label="Menu"
            @click="toggleLeftDrawer"
          />
          <q-toolbar-title class="flex items-center justify-center">
             <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center q-mr-sm">
              <span class="text-white">🧬</span>
            </div>
          </q-toolbar-title>
          <q-btn flat round dense icon="notifications" />
        </q-toolbar>
      </q-header>

      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useLotto } from 'src/composables/useLotto'

const { lotteryOptions, selectedLotteryId, currentLottery, selectLottery, loadDraws } = useLotto()
const leftDrawerOpen = ref(false)

const menuItems = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊', to: '/' },
  { id: 'history', label: 'History Data', icon: '📂', to: '/history' },
  { id: 'models', label: 'Models', icon: '🤖', to: '/models' },
  { id: 'settings', label: 'System', icon: '⚙️', to: '/settings' }
]

onMounted(() => {
  loadDraws()
})

function toggleLeftDrawer () {
  leftDrawerOpen.value = !leftDrawerOpen.value
}

function onLotteryChange (val) {
  selectLottery(val.value)
}
</script>

<style lang="scss">
.nav-active {
  background: rgba(168, 85, 247, 0.1) !important;
  color: #a855f7 !important;
  border: 1px solid rgba(168, 85, 247, 0.2);
}

.nav-item {
  transition: all 0.3s ease;
  &:hover {
    background: rgba(255, 255, 255, 0.05);
    color: white;
  }
}

.lottery-select {
  .q-field__control {
    border-radius: 12px !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    &:hover {
      border-color: #3b82f6 !important;
    }
  }
}
</style>
