<template>

  <router-view />
</template>

<script setup>
import { ref, onErrorCaptured } from 'vue'

const version = ref('2.1')

onErrorCaptured((err, instance, info) => {
  console.error('[Global Error Boundary]:', err)
  console.error('[Component]:', instance)
  console.error('[Info]:', info)
  // Check if we can show it on screen
  const el = document.getElementById('debug-boot')
  if (el) el.innerText = `ERROR: ${err.message}`
  return false // prevent propagation
})

console.log('App.vue mounted')
</script>
