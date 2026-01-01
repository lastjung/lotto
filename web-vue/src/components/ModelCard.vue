<template>
  <q-card
    class="model-card cursor-pointer transition-all"
    :class="{ 'selected': isSelected }"
    flat
    bordered
    @click="$emit('select', model.id)"
  >
    <q-card-section class="text-center">
      <div class="text-3xl mb-2">{{ model.icon }}</div>
      <div class="text-h6 text-weight-bold">{{ model.name }}</div>
      <div class="text-caption text-grey-6">{{ model.description }}</div>
    </q-card-section>

    <!-- Selected Indicator -->
    <div v-if="isSelected" class="selected-indicator">
      <q-icon name="check_circle" color="primary" size="sm" />
    </div>
  </q-card>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  model: {
    type: Object,
    required: true
    // { id, name, icon, description }
  },
  selectedModelId: {
    type: String,
    default: ''
  }
})

defineEmits(['select'])

const isSelected = computed(() => props.model.id === props.selectedModelId)
</script>

<style scoped lang="scss">
.model-card {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  position: relative;
  overflow: hidden;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
  }

  &.selected {
    border-color: var(--q-primary);
    background: rgba(var(--q-primary-rgb), 0.15);
    box-shadow: 0 0 20px rgba(var(--q-primary-rgb), 0.3);
  }
}

.selected-indicator {
  position: absolute;
  top: 8px;
  right: 8px;
}
</style>
