import { ref } from 'vue'
import * as ort from 'onnxruntime-web'

// ORT 환경 설정: public/wasm 폴더에서 로드하도록 명시
ort.env.wasm.wasmPaths = window.location.origin + '/wasm/'

export function useAiEngine() {
    const session = ref(null)
    const modelLoaded = ref(false)
    const loading = ref(false)
    const error = ref(null)

    /**
     * 모델 로드 (복권 종류 및 모델 타입 기반)
     */
    async function loadModel(lotteryId, modelId) {
        // 통계 모델(JS 구현)인 경우 세션 로드 skip
        if (['vector', 'hot_trend', 'cold_theory', 'physics_bias', 'balanced_mix'].includes(modelId)) {
            session.value = null
            modelLoaded.value = true
            return
        }

        loading.value = true
        error.value = null
        modelLoaded.value = false

        try {
            // [Convention] 모델 파일명: {lotteryId}_{modelId}.onnx (예: korea_645_lstm.onnx)
            // 폴백: {modelId}.onnx (구버전 호환)
            const modelPaths = [
                window.location.origin + `/models/${lotteryId}_${modelId}.onnx`,
                window.location.origin + `/models/${modelId}.onnx`
            ]

            let loaded = false
            for (const path of modelPaths) {
                try {
                    console.log(`Trying to load model: ${path}`)
                    session.value = await ort.InferenceSession.create(path)
                    loaded = true
                    break
                } catch (err) {
                    console.warn(`Failed to load from ${path}, trying next...: ${err.message}`)
                }
            }

            if (!loaded) {
                console.warn('⚠️ No ONNX model found. Statistical fallbacks will be used.')
                // Don't throw, just leave session null
            } else {
                modelLoaded.value = true
                console.log(`✅ Loaded ONNX model: ${modelId} for ${lotteryId}`)
            }
        } catch (e) {
            console.error('❌ Model Engine Init Error:', e)
            error.value = e.message
        } finally {
            loading.value = false
        }
    }

    /**
     * AI 생성 (ONNX Inference)
     */
    async function generateWithAi(lottery, pastDraws) {
        if (!session.value) throw new Error('AI Session not loaded')

        // 1. 입력 데이터 준비 (마지막 10회차)
        const windowSize = 10
        const pickCount = lottery.pickCount
        const inputData = new BigInt64Array(windowSize * pickCount).fill(0n)

        const recent = pastDraws.slice(0, windowSize)
        for (let i = 0; i < windowSize; i++) {
            const draw = recent[i] || { numbers: [] }
            const nums = draw.numbers || []
            for (let j = 0; j < pickCount; j++) {
                inputData[i * pickCount + j] = BigInt(nums[j] || 0)
            }
        }

        const inputTensor = new ort.Tensor('int64', inputData, [1, windowSize, pickCount])

        // 2. 실행
        const outputs = await session.value.run({ input: inputTensor })
        const logits = outputs.output.data

        // 3. 샘플링 (여러 세트 생성)
        const results = []
        for (let i = 0; i < 15; i++) {
            const temp = 1.0 + (i * 0.1)
            const numbers = sampleFromLogits(logits, lottery.maxNum, pickCount, temp)
            results.push(numbers)
        }
        return results
    }

    /**
     * 통계 모델 생성 (JS 구현)
     */
    async function generateWithStat(modelId, lottery, pastDraws) {
        // TODO: 포팅 작업 (Physics Bias, Cold Theory 등)
        // 현재는 Mock 구현
        return Array.from({ length: 15 }, () =>
            Array.from({ length: lottery.pickCount }, () => Math.floor(Math.random() * lottery.maxNum) + 1).sort((a, b) => a - b)
        )
    }

    // --- Helper Functions (From original app.js) ---

    function sampleFromLogits(logits, maxNum, pickCount, temperature = 1.0) {
        const numbers = []
        const used = new Set()

        for (let pos = 0; pos < pickCount; pos++) {
            const offset = pos * maxNum
            // Note: Indexing 0-based for AI probabilities
            const slice = logits.slice(offset, offset + maxNum)
            const probs = softmax(slice, temperature)

            let selected = -1
            let attempts = 0
            while (selected === -1 || used.has(selected)) {
                selected = sample(probs) + 1
                attempts++
                if (attempts > 50) break
            }

            if (selected > 0 && selected <= maxNum) {
                used.add(selected)
                numbers.push(selected)
            }
        }

        while (numbers.length < pickCount) {
            const n = Math.floor(Math.random() * maxNum) + 1
            if (!used.has(n)) {
                used.add(n)
                numbers.push(n)
            }
        }
        return numbers.sort((a, b) => a - b)
    }

    function softmax(arr, temperature = 1.0) {
        const scaled = Array.from(arr).map(x => x / temperature)
        const max = Math.max(...scaled)
        const exps = scaled.map(x => Math.exp(x - max))
        const sum = exps.reduce((a, b) => a + b, 0)
        return exps.map(x => x / sum)
    }

    function sample(probs) {
        const r = Math.random()
        let cum = 0
        for (let i = 0; i < probs.length; i++) {
            cum += probs[i]
            if (r < cum) return i
        }
        return probs.length - 1
    }

    return {
        modelLoaded,
        loading,
        error,
        loadModel,
        generateWithAi,
        generateWithStat
    }
}
