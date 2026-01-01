/**
 * AI 로또 분석기 - ONNX 브라우저 추론
 * PyTorch 모델을 ONNX로 변환하여 브라우저에서 직접 실행
 */

// 설정 및 환경 감지
const API_PORT = '8000'; // FastAPI 서버 기본 포트
// [FIX] 포트가 8000이거나 로컬호스트가 아니면 API 모드 사용 (기존 코드 보존)
const IS_STATIC_MODE = window.location.port !== '8000' && window.location.port !== '';
const API_BASE = IS_STATIC_MODE ? '' : 'http://localhost:8000';

// [Persistence] 초기 로드 시 LocalStorage 값 우선 사용
let savedLottery = localStorage.getItem('s_lottery') || 'korea_645';
let savedModel = localStorage.getItem('s_model') || 'transformer';

// [Config] Supabase (from ui.js)
const SB_URL = 'https://sfqlshdlqwqlkxdrfdke.supabase.co';
const SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNmcWxzaGRscXdxbGt4ZHJmZGtlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5MDM0NzUsImV4cCI6MjA4MTQ3OTQ3NX0.CMbJ_5IUxAifoNIzqdxu_3sz31AtOMw2vRBPxfxZzSk';
let supabaseClient = null; // [FIX] Renamed to avoid conflict with window.supabase from CDN

let currentModel = savedModel;
let session = null;
let lottoData = null;
let modelLoaded = false;

// 초기화
document.addEventListener('DOMContentLoaded', async () => {
    console.log(`🚀 AI 로또 분석기 시작 (모드: ${IS_STATIC_MODE ? 'STATIC/ONNX' : 'API/SERVER'})`);

    // [Init] Supabase
    if (window.supabase) {
        try {
            supabaseClient = window.supabase.createClient(SB_URL, SB_KEY);
            console.log('✅ Supabase client initialized');
        } catch (e) {
            console.error('❌ Supabase init failed:', e);
        }
    }

    await loadLottoData();
    if (IS_STATIC_MODE) {
        await loadModel('transformer');
    } else {
        modelLoaded = true; // 서버 모드는 항상 준비됨
        const statusEl = document.getElementById('model-status');
        if (statusEl) statusEl.textContent = '✅ API 서버 모드 (FastAPI 연동)';
    }

    // 이벤트 리스너 복구
    const generateBtn = document.getElementById('generateBtn');
    if (generateBtn) generateBtn.addEventListener('click', generateNumbers);

    // Desktop & Mobile Selectors
    const lotterySelectDesktop = document.getElementById('lotterySelectDesktop');
    const lotterySelectMobile = document.getElementById('lotterySelectMobile');
    const lotterySelectOld = document.getElementById('lotterySelect');

    const handler = (e) => onLotteryChange(e.target.value);
    if (lotterySelectDesktop) lotterySelectDesktop.addEventListener('change', handler);
    if (lotterySelectMobile) lotterySelectMobile.addEventListener('change', handler);
    if (lotterySelectOld) lotterySelectOld.addEventListener('change', handler);

    // [Persistence] 저장된 설정 UI에 반영
    if (lotterySelectDesktop) lotterySelectDesktop.value = savedLottery;
    if (lotterySelectMobile) lotterySelectMobile.value = savedLottery;
    if (lotterySelectOld) lotterySelectOld.value = savedLottery;

    // 복원된 값으로 초기 데이터 로드 (model은 아래에서 로드됨)
    await loadLottoData(savedLottery);

    // 모델 선택 UI 반영 (버튼 활성화)
    selectModel(savedModel, true); // [Refine] 초기 로드 시 자동 생성 방지

    loadHistory(); // 이력 로드
});

// 탭 전환
function switchTab(tabId) {
    // Hide all views
    ['dashboard', 'history', 'models', 'settings'].forEach(tab => {
        const view = document.getElementById(`view-${tab}`);
        if (view) view.classList.add('hidden');
    });

    // Show target view
    const targetView = document.getElementById(`view-${tabId}`);
    if (targetView) targetView.classList.remove('hidden');

    // Update nav button styles
    ['dashboard', 'history', 'models', 'settings'].forEach(tab => {
        const navBtn = document.getElementById(`nav-${tab}`);
        if (!navBtn) return;

        if (tab === tabId) {
            navBtn.classList.remove('text-gray-400');
            navBtn.classList.add('bg-blue-600/10', 'text-blue-400', 'border', 'border-blue-500/20');
        } else {
            navBtn.classList.remove('bg-blue-600/10', 'text-blue-400', 'border', 'border-blue-500/20');
            navBtn.classList.add('text-gray-400');
        }
    });

    if (tabId === 'history') loadHistory();
}

// 로또 데이터 로드
async function loadLottoData(lotteryId = 'korea_645') {
    const dataStatus = document.getElementById('data-status');
    if (dataStatus) dataStatus.textContent = `📡 ${lotteryId} 데이터 불러오는 중...`;

    try {
        const res = await fetch(`data/${lotteryId}/draws.json`);
        if (!res.ok) throw new Error('파일을 찾을 수 없습니다.');

        const json = await res.json();
        lottoData = json.draws || json;

        if (dataStatus) dataStatus.textContent = `✅ ${lotteryId} 데이터 로드 완료 (${lottoData.length}회차)`;
        console.log(`✅ 로또 데이터 로드 (${lotteryId}): ${lottoData.length}회차`);
    } catch (e) {
        console.error('❌ 데이터 로드 실패:', e);
        if (dataStatus) dataStatus.textContent = `❌ ${lotteryId} 데이터 로드 실패 (파일 확인 필요)`;
    }
}

// [Helper] 서버로 설정 저장 (Dual Save)
async function saveConfigToServer(lottery, model) {
    if (IS_STATIC_MODE) return;
    try {
        await fetch(`${API_BASE}/api/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                default_lottery: lottery,
                default_model: model,
                updated_at: new Date().toISOString()
            })
        });
        console.log("✅ Config saved to SERVER");
    } catch (e) {
        console.error("❌ Failed to save config to server:", e);
    }
}

// [Helper] 안전하게 복권 값 가져오기
function getLotteryValue() {
    const desktop = document.getElementById('lotterySelectDesktop');
    const mobile = document.getElementById('lotterySelectMobile');
    const old = document.getElementById('lotterySelect');

    if (desktop && desktop.value) return desktop.value;
    if (mobile && mobile.value) return mobile.value;
    if (old && old.value) return old.value;

    return 'korea_645'; // Default fallback
}

// 복권 종류 변경 처리
async function onLotteryChange() {
    const lotteryId = getLotteryValue(); // [FIX] Use helper

    // [Persistence] 1. LocalStorage 저장
    localStorage.setItem('s_lottery', lotteryId);

    await loadLottoData(lotteryId);

    // [Persistence] 2. Config 저장 (모델 변경 시와 동일하게)
    // 현재는 모델 변경 시에만 config 저장이 트리거되므로, 
    // 여기서는 간단히 로컬 변수 업데이트만 하고, 실제 저장은 selectModel이나 생성 시점에 될 수 있음
    // 하지만 "두 군데 저장" 요구사항에 맞춰 즉시 저장 시도
    saveConfigToServer(lotteryId, currentModel);

    // 모델도 해당 복권에 맞춰 다시 로딩 (나중에 국가별 모델이 다를 경우 대비)
    await loadModel(currentModel);
}

// ONNX 모델 로드
async function loadModel(modelType) {
    // API 모드일 경우 클라이언트 모델 로딩 건너뜀
    if (!IS_STATIC_MODE) {
        modelLoaded = true;
        const statusEl = document.getElementById('model-status');
        if (statusEl) statusEl.textContent = `✅ ${modelType.toUpperCase()} (API Mode)`;
        console.log(`ℹ️ Model selection updated to ${modelType} (Server-side)`);
        return;
    }

    const statusEl = document.getElementById('model-status');
    if (statusEl) statusEl.textContent = `📦 ${modelType.toUpperCase()} 모델 로딩 중...`;

    try {
        if (modelType === 'vector' || modelType === 'hot_trend') {
            // Vector/Hot Trend는 JS로 구현 (ONNX 없음)
            modelLoaded = true;
            if (statusEl) statusEl.textContent = `✅ ${modelType.toUpperCase()} 준비 완료 (JS 구현)`;
            return;
        }

        session = await ort.InferenceSession.create(`models/${modelType}.onnx`);
        modelLoaded = true;
        if (statusEl) statusEl.textContent = `✅ ${modelType.toUpperCase()} 모델 로드 완료`;
        console.log(`✅ ONNX 모델 로드: ${modelType}`);
    } catch (e) {
        console.error('❌ 모델 로드 실패:', e);
        if (statusEl) statusEl.textContent = `❌ 모델 로드 실패: ${e.message}`;
        modelLoaded = false;
    }
}


// 모델 선택
async function selectModel(type, isInit = false) {
    currentModel = type;

    // 버튼 스타일 업데이트
    ['transformer', 'lstm', 'vector', 'hot_trend'].forEach(m => {
        const btn = document.getElementById(`btn-${m}`);
        // 구버전/신버전 ID 호환성 체크 (btn- vs card-)
        const cardBtn = document.getElementById(`card-${m}`);
        const target = btn || cardBtn;

        if (!target) return; // 요소가 없으면 스킵

        if (m === type) {
            target.classList.add('border-purple-500', 'bg-purple-500/20', 'text-white');
            target.classList.remove('border-gray-700', 'bg-gray-800', 'text-gray-400');
        } else {
            target.classList.remove('border-purple-500', 'bg-purple-500/20', 'text-white');
            target.classList.add('border-gray-700', 'bg-gray-800', 'text-gray-400');
        }
    });

    // [Persistence] 1. LocalStorage 저장
    if (!isInit) localStorage.setItem('s_model', type);

    await loadModel(type);

    // [특수 기능] 모든 모델 자동 실행 (초기 로드 시에는 실행 안 함)
    if (!isInit && ['transformer', 'lstm', 'vector', 'hot_trend'].includes(type)) {
        console.log(`⚡ ${type} Card Clicked: Executing Auto-Generate Flow`);

        // [Persistence] 2. Config 저장 (서버) - Dual Save
        await saveConfigToServer(getLotteryValue(), type);

        // 2. 번호 생성 (DB 저장 및 결과 표시는 generateNumbers 내부에서 처리됨)
        await generateNumbers();
    }
}
// [Integrate] UI.js와의 호환성을 위해 전역 노출
window.appSelectModel = selectModel;


// 번호 생성
async function generateNumbers() {
    if (IS_STATIC_MODE && !modelLoaded) {
        alert('모델이 로드되지 않았습니다.');
        return;
    }

    const loading = document.getElementById('loading') || document.getElementById('progressArea');
    const results = document.getElementById('numbersArea') || document.getElementById('resultsArea');

    if (loading) loading.classList.remove('hidden');
    if (results) results.innerHTML = '';

    try {
        let generated_data;

        if (IS_STATIC_MODE) {
            // --- ONNX 모드 (8081) ---
            let raw_numbers;
            if (currentModel === 'vector') {
                raw_numbers = await generateWithVector();
            } else if (currentModel === 'hot_trend') {
                raw_numbers = await generateWithHotTrend();
            } else {
                raw_numbers = await generateWithONNX();
            }
            // 필터 적용 (클라이언트 사이드)
            const filtered = applyFilters(raw_numbers).slice(0, 5);
            generated_data = {
                numbers: filtered.map(nums => ({
                    numbers: nums,
                    analysis: {
                        sum: nums.reduce((a, b) => a + b, 0),
                        ac_value: calculateAC(nums)
                    }
                })),
                lottery_id: getLotteryValue(), // [FIX] Use helper
                model: currentModel,
                generated_at: new Date().toISOString()
            };
        } else {
            // --- API 모드 (8000) ---
            const res = await fetch(`${API_BASE}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lottery_id: getLotteryValue(), // [FIX] Use helper
                    count: 5,
                    model_type: currentModel,
                    ac_filter: document.getElementById('acFilter').checked,
                    sum_filter: document.getElementById('sumFilter').checked,
                    consecutive_filter: document.getElementById('consecutiveFilter').checked
                })
            });
            if (!res.ok) throw new Error('서버 생성 실패');
            generated_data = await res.json();
        }

        // 🎬 Play animation, then show results (supports multiple animation types)
        const animationContainer = document.getElementById('resultsArea') || document.getElementById('numbersArea');

        if (window.animationManager && animationContainer) {
            // Get first set of numbers for animation
            const firstSetNumbers = generated_data.numbers[0]?.numbers || [];

            // Get animation from manager (respects user preference)
            const animation = window.animationManager.getAnimation(() => {
                // After animation, show full results
                setTimeout(() => {
                    displayResults(generated_data);
                    saveHistoryEntry(generated_data);
                }, 500);
            });

            // Run animation with first set numbers
            animation.animate(firstSetNumbers, animationContainer);
        } else if (window.LotteryAnimation && animationContainer) {
            // Fallback to direct LotteryAnimation if no manager
            const firstSetNumbers = generated_data.numbers[0]?.numbers || [];
            if (!window.lottoAnim) {
                window.lottoAnim = new LotteryAnimation({
                    soundEnabled: true,
                    onComplete: () => {
                        setTimeout(() => {
                            displayResults(generated_data);
                            saveHistoryEntry(generated_data);
                        }, 500);
                    }
                });
            }
            window.lottoAnim.animate(firstSetNumbers, animationContainer);
        } else {
            // Fallback: No animation, show directly
            displayResults(generated_data);
            saveHistoryEntry(generated_data);
        }

    } catch (e) {
        console.error('❌ 생성 실패:', e);
        if (results) results.innerHTML = `<p class="text-red-400">생성 실패: ${e.message}</p>`;
    } finally {
        if (loading) loading.classList.add('hidden');
    }
}

// ONNX 모델로 생성
async function generateWithONNX() {
    const recent = getRecentDraws(10);
    const inputData = new BigInt64Array(60).fill(0n); // 0으로 초기화

    // 입력 데이터 준비 (10회차 x 6개 번호)
    for (let i = 0; i < 10; i++) {
        const draw = recent[i] || [];
        for (let j = 0; j < 6; j++) {
            // 번호가 6개보다 적으면 (예: 파워볼 5개) 0으로 채우거나 있는 것만 넣음
            const val = draw[j];
            inputData[i * 6 + j] = val !== undefined ? BigInt(val) : 0n;
        }
    }

    const inputTensor = new ort.Tensor('int64', inputData, [1, 10, 6]);
    const outputs = await session.run({ input: inputTensor });
    const logits = outputs.output.data;

    // 여러 세트 생성
    const generated = [];
    for (let set = 0; set < 15; set++) {
        const numbers = sampleFromLogits(logits, 1.0 + set * 0.1);
        generated.push(numbers);
    }

    return generated;
}

// Hot Trend (최근 빈도 기반 가중치) 생성
async function generateWithHotTrend() {
    const generated = [];
    const recentDraws = getRecentDraws(30); // 최근 30회차 분석
    const frequency = new Array(46).fill(0);

    // 빈도 분석
    recentDraws.forEach(draw => {
        draw.forEach(num => frequency[num]++);
    });

    // 가중치 기반 랜덤 선택 (Weighted Random)
    for (let i = 0; i < 5; i++) { // 5게임 생성
        const numbers = new Set();
        while (numbers.size < 6) {
            // 룰렛 휠 선택 방식 (Roulette Wheel Selection)
            let totalWeight = 0;
            // 기본 가중치 1 + 빈도 * 2 (빈도 높은 번호 우대)
            const weights = frequency.map(f => 1 + (f * 2));
            weights[0] = 0; // 0번 인덱스 제외

            weights.forEach(w => totalWeight += w);
            let randomVal = Math.random() * totalWeight;

            for (let n = 1; n <= 45; n++) {
                randomVal -= weights[n];
                if (randomVal <= 0) {
                    if (!numbers.has(n)) numbers.add(n);
                    break;
                }
            }
        }
        generated.push([...numbers].sort((a, b) => a - b));
    }
    return generated;
}

// Vector 모델로 생성 (순수 JS 구현)
async function generateWithVector() {
    const generated = [];
    const allNumbers = lottoData.map(d => d.numbers);

    for (let i = 0; i < 15; i++) {
        // 랜덤하게 이전 당첨 번호 조합
        const indices = [];
        while (indices.length < 3) {
            const idx = Math.floor(Math.random() * allNumbers.length);
            if (!indices.includes(idx)) indices.push(idx);
        }

        // 번호 빈도 계산
        const freq = new Array(46).fill(0);
        indices.forEach(idx => {
            allNumbers[idx].forEach(n => freq[n]++);
        });

        // 상위 빈도 + 랜덤 조합
        const candidates = [];
        for (let n = 1; n <= 45; n++) {
            candidates.push({ num: n, freq: freq[n] + Math.random() });
        }
        candidates.sort((a, b) => b.freq - a.freq);

        const numbers = candidates.slice(0, 6).map(c => c.num).sort((a, b) => a - b);
        generated.push(numbers);
    }

    return generated;
}

// 로짓에서 샘플링
function sampleFromLogits(logits, temperature = 1.0) {
    const numbers = [];
    const used = new Set();

    // 6개 위치 각각에서 샘플링
    for (let pos = 0; pos < 6; pos++) {
        const offset = pos * 45;
        const probs = softmax(logits.slice(offset, offset + 45), temperature);

        let selected = -1;
        let attempts = 0;
        while (selected === -1 || used.has(selected)) {
            selected = sample(probs) + 1;
            attempts++;
            if (attempts > 100) break;
        }

        if (selected > 0 && selected <= 45) {
            used.add(selected);
            numbers.push(selected);
        }
    }

    // 6개 미만이면 랜덤 채우기
    while (numbers.length < 6) {
        const n = Math.floor(Math.random() * 45) + 1;
        if (!used.has(n)) {
            used.add(n);
            numbers.push(n);
        }
    }

    return numbers.sort((a, b) => a - b);
}

// Softmax
function softmax(arr, temperature = 1.0) {
    const scaled = arr.map(x => x / temperature);
    const max = Math.max(...scaled);
    const exps = scaled.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// 확률 분포에서 샘플링
function sample(probs) {
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return probs.length - 1;
}

// 최근 N회차 데이터
function getRecentDraws(n) {
    return lottoData.slice(-n).map(d => d.numbers);
}

// 필터 적용
function applyFilters(numbersList) {
    const acFilter = document.getElementById('acFilter').checked;
    const sumFilter = document.getElementById('sumFilter').checked;
    const consecutiveFilter = document.getElementById('consecutiveFilter').checked;

    return numbersList.filter(numbers => {
        // AC 필터
        if (acFilter && calculateAC(numbers) < 7) return false;

        // 합계 필터
        if (sumFilter) {
            const sum = numbers.reduce((a, b) => a + b, 0);
            if (sum < 100 || sum > 175) return false;
        }

        // 연속번호 필터
        if (consecutiveFilter && hasConsecutive(numbers)) return false;

        return true;
    });
}

// AC값 계산
function calculateAC(numbers) {
    const sorted = [...numbers].sort((a, b) => a - b);
    const diffs = new Set();

    for (let i = 0; i < sorted.length; i++) {
        for (let j = i + 1; j < sorted.length; j++) {
            diffs.add(sorted[j] - sorted[i]);
        }
    }

    return diffs.size - (numbers.length - 1);
}

// 연속번호 검사
function hasConsecutive(numbers, minCount = 3) {
    const sorted = [...numbers].sort((a, b) => a - b);
    let consecutive = 1;

    for (let i = 1; i < sorted.length; i++) {
        if (sorted[i] - sorted[i - 1] === 1) {
            consecutive++;
            if (consecutive >= minCount) return true;
        } else {
            consecutive = 1;
        }
    }
    return false;
}

// 결과 표시 (UI)
function displayResults(data) {
    const area = document.getElementById('resultsArea') || document.getElementById('numbersArea');
    if (!area) return;

    // Hide placeholder when results are shown
    const placeholder = document.getElementById('resultsPlaceholder');
    if (placeholder) placeholder.classList.add('hidden');

    if (!data.numbers || data.numbers.length === 0) {
        area.innerHTML = '<p class="text-yellow-400">조건에 맞는 번호가 없습니다. 필터를 조정해주세요.</p>';
        return;
    }

    area.innerHTML = `
        <div class="flex items-center justify-between mb-4">
            <div class="text-sm text-gray-400">
                <span class="text-blue-400 font-bold mr-1">
                    ${(function () {
            const sel = document.getElementById('lotterySelectDesktop') || document.getElementById('lotterySelectMobile') || document.getElementById('lotterySelect');
            return sel && sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text.trim() : 'Korea Lotto 6/45';
        })()}
                </span>
                | <span class="text-purple-400 font-bold">${data.model ? data.model.toUpperCase() : 'AI'}</span>
                | Draw ${data.target_draw || 'Next'}
            </div>
            <span class="text-xs text-gray-600">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="space-y-3">
        ${data.numbers.map((item, i) => {
            const nums = item.numbers;
            const analysis = item.analysis || {};

            return `
            <div class="bg-white/5 rounded-2xl p-4 border border-white/5 hover:bg-white/10 transition-all group">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-1 rounded">SET #${i + 1}</span>
                    <div class="flex gap-2 text-[10px] text-gray-500">
                        <span>Sum: <b class="text-gray-300">${analysis.sum}</b></span>
                        <span>AC: <b class="text-gray-300">${analysis.ac_value}</b></span>
                    </div>
                </div>
                <div class="flex gap-2 justify-center">
                    ${nums.map(n => `
                        <span class="lotto-ball-v2 ${getBallClass(n)} pop-in" style="animation-delay: ${i * 0.1}s">
                            ${n}
                        </span>
                    `).join('')}
                </div>
            </div>
        `;
        }).join('')}
    </div>`;
}

// [Persistence] Cloud Save (Supabase)
async function saveToSupabase(data) {
    if (!supabaseClient) return;

    try {
        const payload = {
            created_at: new Date().toISOString(),
            numbers: data.numbers,
            model: data.model || currentModel,
            lottery_type: getLotteryValue(),
            // Simple user ID from localStorage or generate new
            user_id: localStorage.getItem('lotto_user_id') || (() => {
                const id = 'user_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('lotto_user_id', id);
                return id;
            })()
        };

        const { error } = await supabaseClient.from('lotto_history').insert([payload]);
        if (error) console.warn('❌ Supabase save error:', error.message);
        else console.log('✅ Saved to Supabase DB');
    } catch (e) {
        console.warn('❌ Supabase network error:', e);
    }
}

// 이력 저장 (공통)
function saveHistoryEntry(data) {
    try {
        // 1. LocalStorage
        const history = JSON.parse(localStorage.getItem('lotto_history') || '[]');
        const entry = {
            id: Date.now(),
            date: new Date().toISOString(),
            model: data.model || currentModel,
            lottery_type: getLotteryValue(),
            lottery_name: (function () {
                const sel = document.getElementById('lotterySelectDesktop') || document.getElementById('lotterySelectMobile') || document.getElementById('lotterySelect');
                return sel && sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text.trim() : 'Korea Lotto 6/45';
            })(),
            numbers: data.numbers || [],
            generated_at: new Date().toISOString()
        };

        history.unshift(entry);
        const limitedHistory = history.slice(0, 100);
        localStorage.setItem('lotto_history', JSON.stringify(limitedHistory));
        console.log("✅ Local History saved");

        // 2. Cloud DB (Supabase)
        saveToSupabase(data);

    } catch (e) {
        console.error("❌ Failed to save history:", e);
    }
}

// 이력 로드 및 표시 (Robust for Null/Error)
function loadHistory() {
    const area = document.getElementById('historyArea');
    if (!area) return;

    let history = [];
    try {
        const raw = localStorage.getItem('lotto_history');
        if (raw) {
            history = JSON.parse(raw);
            if (!Array.isArray(history)) history = [];
        }
    } catch (e) {
        console.error("Local History Corrupted, resetting:", e);
        localStorage.removeItem('lotto_history');
        history = [];
    }

    if (history.length === 0) {
        area.innerHTML = '<div class="text-center text-gray-500 py-10">No history data available.</div>';
        return;
    }

    area.innerHTML = history.map((entry, idx) => {
        if (!entry || !entry.numbers) return ''; // Skip invalid entries

        // Handle different data structures (array of arrays vs array of objects)
        const numberSets = Array.isArray(entry.numbers) ? entry.numbers : [];
        const modelName = entry.model || 'Unknown';

        return `
        <div class="glass-panel p-4 rounded-xl border border-white/10 mb-4">
            <div class="flex justify-between items-start mb-3 border-b border-white/5 pb-2">
                <div class="text-xs text-gray-400">
                    ${new Date(entry.date || entry.generated_at || Date.now()).toLocaleString()}
                </div>
                <div class="flex gap-2">
                    <span class="bg-green-500/20 text-green-300 px-2 py-0.5 rounded text-[10px] font-bold">${entry.lottery_name || 'Korea Lotto'}</span>
                    <span class="bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded text-[10px] font-bold">${modelName.toUpperCase()}</span>
                </div>
            </div>
            <div class="space-y-2">
                ${numberSets.map(set => {
            // entry.numbers structure check: could be [1,2,3..] or {numbers:[1,2..]}
            const nums = Array.isArray(set) ? set : (set.numbers || []);
            if (nums.length === 0) return '';
            return `
                    <div class="flex gap-1.5 flex-wrap justify-center sm:justify-start">
                        ${nums.map(n => `<span class="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold ${getBallColor(n)}">${n}</span>`).join('')}
                    </div>
                    `;
        }).join('')}
            </div>
        </div>
    `}).join('');
}

function clearHistory() {
    if (confirm('Clear all local history?')) {
        localStorage.removeItem('lotto_history');
        loadHistory();
    }
}

// 공 색상 클래스 (V2 대응)
function getBallClass(num) {
    if (num <= 10) return 'ball-1-10';
    if (num <= 20) return 'ball-11-20';
    if (num <= 30) return 'ball-21-30';
    if (num <= 40) return 'ball-31-40';
    return 'ball-41-45';
}

function getBallColor(n) {
    // Fallback for missing CSS classes or history view
    if (n <= 10) return 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20';
    if (n <= 20) return 'bg-blue-500 text-white shadow-lg shadow-blue-500/20';
    if (n <= 30) return 'bg-red-500 text-white shadow-lg shadow-red-500/20';
    if (n <= 40) return 'bg-gray-600 text-white shadow-lg shadow-gray-500/20';
    return 'bg-green-500 text-white shadow-lg shadow-green-500/20';
}

