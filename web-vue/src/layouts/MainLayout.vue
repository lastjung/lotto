<template>
  <q-layout view="lHh LpR fFf" class="bg-[#0f172a] text-white selection:bg-blue-500/30 font-sans">
    <!-- Fixed Background Gradient (Matches Port 8000 exactly) -->
    <div class="fixed inset-0 pointer-events-none z-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-[#0f172a] to-[#0f172a]"></div>
    <q-drawer
      v-model="leftDrawerOpen"
      show-if-above
      :width="256"
      class="sidebar-glass text-white shadow-24"
    >
      <div class="column h-full overflow-hidden">
        <!-- Logo Section (Match web/index.html: p-6 pb-2) -->
        <div class="p-6 pb-2 flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
             <span class="text-white text-lg">🧬</span>
          </div>
          <h1 class="font-bold text-xl text-white tracking-tight heading-font">LottoQuant<span class="text-blue-400">AI</span></h1>
        </div>

        <!-- Lottery Selector (Match web/index.html: Native Select) -->
        <div class="px-4 mb-6 mt-4">
            <div class="relative group">
                <select 
                    v-model="selectedLotteryId"
                    @change="onLotteryChange"
                    class="w-full bg-[#0f172a] border border-blue-500/30 text-white text-sm font-bold rounded-xl px-4 py-3 appearance-none cursor-pointer hover:border-blue-400 transition-colors focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                    <option v-for="lottery in lotteryOptions" :key="lottery.id" :value="lottery.id">
                        {{ lottery.name }}
                    </option>
                </select>
                <!-- Custom Arrow -->
                <span class="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-blue-400 pointer-events-none">▼</span>
            </div>
        </div>

        <!-- Navigation (Match web/index.html: proper spacing and native buttons/links) -->
        <nav class="flex-1 px-4 py-2 space-y-2">
          <router-link 
            v-for="item in menuItems" 
            :key="item.id" 
            :to="item.to"
            custom
            v-slot="{ navigate, isActive }"
          >
            <button 
                @click="navigate" 
                class="w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-left"
                :class="isActive ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20 hover:bg-blue-600/20' : 'text-gray-400 hover:text-white hover:bg-white/5'"
            >
                <span class="text-lg">{{ item.icon }}</span>
                <span class="font-bold text-lg">{{ item.label }}</span>
            </button>
          </router-link>
        </nav>

        <!-- System Status Box (Match web/index.html) -->
        <div class="p-4 border-t border-white/5">
          <div class="bg-black/20 rounded-xl p-4">
            <div class="flex items-center gap-2 mb-2">
                <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                <span class="text-xs font-bold text-green-400 uppercase">System Online</span>
            </div>
            <div class="text-[10px] text-gray-500">
                Engine: <span class="text-gray-300">v2.1.0 (ONNX)</span><br>
                Agent: <span class="text-gray-300">Connected</span>
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
import { onMounted } from 'vue'
import { useLotto } from 'src/composables/useLotto'

const {
  lotteryOptions,
  selectedLotteryId,
  selectLottery,
  loadDraws
} = useLotto()

const menuItems = [
  { id: 'dash', label: 'Dashboard', icon: '📊', to: '/' },
  { id: 'history', label: 'History Analysis', icon: '📂', to: '/history' },
  { id: 'models', label: 'Models', icon: '🤖', to: '/models' },
  { id: 'system', label: 'System', icon: '⚙️', to: '/system' }
]

onMounted(() => {
  loadDraws()
})

function onLotteryChange (event) {
  selectLottery(selectedLotteryId.value)
  window.location.reload()
}
</script>


<style lang="scss">
/* Global override to prevent Quasar's default white background on drawers from muddying the color */
.q-drawer { background: transparent !important; }

.sidebar-glass {
  background: rgba(30, 41, 59, 0.5) !important; /* Reverted to exactly match legacy (50% opacity) */
  backdrop-filter: blur(16px);
  border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.nav-active {
  background: rgba(30, 58, 138, 0.5) !important;
  color: #fff !important;
  border: 1px solid rgba(59, 130, 246, 0.3);
  box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
  .q-item__label {
    text-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
  }
  .q-icon {
    color: white !important;
    filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.5));
  }
}

.nav-item {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  &:hover {
    background: rgba(255, 255, 255, 0.05);
    color: white;
  }
}

.lottery-select-custom {
  .q-field__control {
    border-radius: 20px !important;
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 0 16px !important;
    &:before, &:after { display: none; }
    &:hover { border-color: rgba(59, 130, 246, 0.4); }
  }
  .q-field__native {
    color: white !important;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: -0.025em;
  }
}
</style>
