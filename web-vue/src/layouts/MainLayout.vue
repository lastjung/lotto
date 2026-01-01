<template>
  <q-layout view="hHh lpR fFf" class="bg-dark text-white">
    <q-header elevated class="bg-dark-page">
      <q-toolbar>
        <q-toolbar-title class="flex items-center gap-2">
          <span class="text-h6">🎱 AI Lotto</span>
        </q-toolbar-title>

        <q-space />

        <!-- GitHub Link -->
        <q-btn
          flat
          round
          dense
          icon="mdi-github"
          href="https://github.com/lastjung/lotto"
          target="_blank"
        />
      </q-toolbar>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>

    <q-footer class="bg-transparent text-center q-pa-sm">
      <span class="text-grey-7 text-caption">
        © 2025 AI Lotto Analyzer • Powered by Deep Learning
      </span>
    </q-footer>
  </q-layout>
</template>

<script setup>
</script>

<style lang="scss">
.bg-dark-page {
  background: rgba(15, 15, 25, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
