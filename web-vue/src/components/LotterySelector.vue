<template>
  <q-select
    v-model="selected"
    :options="options"
    option-value="id"
    option-label="name"
    emit-value
    map-options
    dark
    outlined
    dense
    class="lottery-selector"
    dropdown-icon="expand_more"
    @update:model-value="$emit('change', $event)"
  >
    <template #prepend>
      <q-icon name="casino" color="green-4" />
    </template>

    <template #option="{ opt, itemProps }">
      <q-item v-bind="itemProps" class="bg-dark">
        <q-item-section>
          <q-item-label>{{ opt.name }}</q-item-label>
          <q-item-label caption>
            {{ opt.pickCount }} numbers from 1-{{ opt.maxNum }}
            {{ opt.bonus ? `+ Bonus 1-${opt.bonus}` : '' }}
          </q-item-label>
        </q-item-section>
      </q-item>
    </template>
  </q-select>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: String,
    default: 'korea_645'
  },
  options: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['update:modelValue', 'change'])

const selected = ref(props.modelValue)

watch(() => props.modelValue, (val) => {
  selected.value = val
})

watch(selected, (val) => {
  emit('update:modelValue', val)
})
</script>

<style scoped lang="scss">
.lottery-selector {
  min-width: 200px;

  :deep(.q-field__control) {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
  }
}
</style>
