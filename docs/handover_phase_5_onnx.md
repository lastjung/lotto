# 🎱 Phase 5 Handover: ONNX AI Engine Integration & Bug Fixes

> [!IMPORTANT]
> **독립적 개발 원칙**: 이 `web-vue` 프로젝트는 기존 `web` 또는 `web-static` 폴더와 완전히 **독립적으로** 개발되어야 합니다. 기존 레거시 코드에 의존하지 않고, Quasar + Supabase + ONNX (Client-side) 기반의 새로운 아키텍처를 유지하는 것이 핵심입니다.

## 📌 Current Status
- **Phase 1-4**: 100% Completed (Layout, Supabase, Charts, Real-time Data).
- **Phase 5 (ONNX)**: Core logic implemented in `useAiEngine.js`, but currently facing a **Blank Page / Asset Loading** issue in the local dev environment.

## 🛠 Actions Taken & Technical Details

### 1. ONNX & WASM Support
- Created `useAiEngine.js` for client-side inference.
- Installed `onnxruntime-web`.
- **WASM Fix**: Copied `.wasm` and `.mjs` files from `node_modules` to `public/wasm/`.
- **MIME/Vite Fix**: Renamed all `public/wasm/*.mjs` files to `*.js`. This is critical because Vite attempts to transform `.mjs` files in `public/` as source modules, leading to 500 errors and worker crashes.
- **Config**: Updated `quasar.config.js` to include COOP/COEP headers and `assetsInclude` for `.onnx`, `.wasm`, `.js`.

### 2. Data Integration
- Populated `public/data/` with historical JSON draws from the root folder.
- `useLotto.js` successfully fetches 1200+ draws (`korea_645`) for the charts.
- SVG Charts in `LottoCharts.vue` are fully functional and verified.

## 🔴 Critical Issue: Blank Page at http://localhost:9000 (UNRESOLVED)
- **Status**: ❌ **FAILED** (Vue App not mounting).
- **Symptoms**: `#q-app` div is empty. `client-entry.js` loads (200 OK), but execution halts silently. Even with `onErrorCaptured` and debug overlays in `App.vue`, nothing renders.
- **Attempts Made**:
    1. `optimizeDeps.exclude` for `onnxruntime-web` (to fix Vite worker issues).
    2. Renaming `.mjs` assets to `.js`.
    3. `try-catch` wrapping in `boot/supabase.js` and `IndexPage.vue`.
    4. Cache clearing (`.quasar`, `node_modules/.vite`).
- **Conclusion**: There is a silent failure preventing the Vue instance from mounting. It requires a deeper investigation into the Vite/Quasar bootstrap process or a potential conflict with the `onnxruntime-web` WASM worker interaction that crashes the main thread before Vue can handle errors.

## ⏭ Next Steps for Gemini Pro
1. **Deep Debugging**: 
   - Start by **removing** `onnxruntime-web` and `useAiEngine` entirely to see if the app mounts. Isolate the dependency.
   - Check `client-entry.js` execution flow.
2. **Complete AI Inference**: 
   - Replace the `timeout` in `IndexPage.vue`'s `generate` function with the actual `generateWithAi` call once the session loads.
3. **Statistical Models**: 
   - Port the actual logic for `Vector`, `Hot Trend`, and `Physics Bias` from `web/js/app.js` to `useAiEngine.js` (currently mock).
4. **Mobile Optimization**: 
   - Refine the sidebar/drawer for smaller screens.

## 📂 Key Files
- `web-vue/src/composables/useAiEngine.js`: ONNX core.
- `web-vue/src/pages/IndexPage.vue`: Dashboard controller.
- `web-vue/quasar.config.js`: Vite & Server headers.
- `web-vue/public/wasm/`: Engine assets.
