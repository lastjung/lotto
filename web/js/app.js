/**
 * AI 로또 분석기 - ONNX 브라우저 추론
 * PyTorch 모델을 ONNX로 변환하여 브라우저에서 직접 실행
 */

// 설정 및 환경 감지
const API_PORT = '8000'; // FastAPI 서버 기본 포트
const IS_STATIC_MODE = window.location.port !== API_PORT || window.location.hostname.includes('github.io') || window.location.hostname.includes('vercel.app');
const API_BASE = ''; // 같은 호스트일 경우 비워둠

let currentModel = 'transformer';
let session = null;
let lottoData = null;
let modelLoaded = false;

// 초기화
document.addEventListener('DOMContentLoaded', async () => {
    console.log(`🚀 AI 로또 분석기 시작 (모드: ${IS_STATIC_MODE ? 'STATIC/ONNX' : 'API/SERVER'})`);
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

    const lotterySelect = document.getElementById('lotterySelect');
    if (lotterySelect) lotterySelect.addEventListener('change', onLotteryChange);

    loadHistory(); // 이력 로드
});

// 탭 전환
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.getElementById(`content-${tabId}`).classList.remove('hidden');

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-purple-600', 'shadow-lg', 'text-white');
        btn.classList.add('text-gray-400');
    });
    const activeBtn = document.getElementById(`tab-btn-${tabId}`);
    activeBtn.classList.remove('text-gray-400');
    activeBtn.classList.add('bg-purple-600', 'shadow-lg', 'text-white');

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

// 복권 종류 변경 처리
async function onLotteryChange() {
    const lotteryId = document.getElementById('lotterySelect').value;
    await loadLottoData(lotteryId);

    // 모델도 해당 복권에 맞춰 다시 로딩 (나중에 국가별 모델이 다를 경우 대비)
    await loadModel(currentModel);
}

// ONNX 모델 로드
async function loadModel(modelType) {
    const statusEl = document.getElementById('model-status');
    statusEl.textContent = `📦 ${modelType.toUpperCase()} 모델 로딩 중...`;

    try {
        if (modelType === 'vector') {
            // Vector는 JS로 구현 (ONNX 없음)
            modelLoaded = true;
            statusEl.textContent = '✅ Vector 모델 준비 완료 (JS 구현)';
            return;
        }

        session = await ort.InferenceSession.create(`models/${modelType}.onnx`);
        modelLoaded = true;
        statusEl.textContent = `✅ ${modelType.toUpperCase()} 모델 로드 완료`;
        console.log(`✅ ONNX 모델 로드: ${modelType}`);
    } catch (e) {
        console.error('❌ 모델 로드 실패:', e);
        statusEl.textContent = `❌ 모델 로드 실패: ${e.message}`;
        modelLoaded = false;
    }
}

// 모델 선택
async function selectModel(type) {
    currentModel = type;

    // 버튼 스타일 업데이트
    ['transformer', 'lstm', 'vector'].forEach(m => {
        const btn = document.getElementById(`btn-${m}`);
        if (m === type) {
            btn.classList.add('border-purple-500', 'bg-purple-500/20', 'text-white');
            btn.classList.remove('border-gray-700', 'bg-gray-800', 'text-gray-400');
        } else {
            btn.classList.remove('border-purple-500', 'bg-purple-500/20', 'text-white');
            btn.classList.add('border-gray-700', 'bg-gray-800', 'text-gray-400');
        }
    });

    await loadModel(type);
}

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
                lottery_id: document.getElementById('lotterySelect').value,
                model: currentModel,
                generated_at: new Date().toISOString()
            };
        } else {
            // --- API 모드 (8000) ---
            const res = await fetch(`${API_BASE}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lottery_id: document.getElementById('lotterySelect').value,
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

        // 결과 표시 및 저장 (LocalStorage 공통 사용)
        displayResults(generated_data);
        saveHistoryEntry(generated_data);

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
    const area = document.getElementById('numbersArea') || document.getElementById('resultsArea');
    if (!area) return;

    if (!data.numbers || data.numbers.length === 0) {
        area.innerHTML = '<p class="text-yellow-400">조건에 맞는 번호가 없습니다. 필터를 조정해주세요.</p>';
        return;
    }

    area.innerHTML = `
        <div class="text-sm text-gray-400 mb-2">
            📅 ${new Date(data.generated_at).toLocaleString('ko-KR')} | 
            <span class="text-purple-400 font-bold">${data.model.toUpperCase()}</span> 모델 |
            🎯 ${data.target_draw || '예측'}회차 대상
        </div>
        ${data.numbers.map((item, i) => {
        const nums = item.numbers;
        const analysis = item.analysis;

        return `
            <div class="bg-black/30 rounded-lg p-4 transition-all hover:bg-black/40 border border-white/5">
                <div class="flex items-center gap-2 mb-2">
                    <span class="text-gray-500 font-mono">#${i + 1}</span>
                    <div class="flex gap-2">
                        ${nums.map(n => `
                            <span class="lotto-ball ${getBallClass(n)} pop-in" style="width:36px; height:36px; font-size:14px;">
                                ${n}
                            </span>
                        `).join('')}
                    </div>
                </div>
                <div class="text-xs md:text-sm text-gray-400 flex flex-wrap gap-3">
                    <span>합계: <strong class="text-gray-200">${analysis.sum}</strong></span>
                    <span>AC: <strong class="text-gray-200">${analysis.ac_value}</strong></span>
                    ${analysis.odd_count !== undefined ? `<span>홀짝: ${analysis.odd_count}:${analysis.even_count}</span>` : ''}
                </div>
            </div>
        `;
    }).join('')}`;
}

// 이력 저장 (공통)
function saveHistoryEntry(data) {
    const history = JSON.parse(localStorage.getItem('lotto_history') || '[]');
    history.unshift(data);
    localStorage.setItem('lotto_history', JSON.stringify(history.slice(0, 1000)));
}

// 이력 저장
function saveToHistory(numbersList) {
    if (numbersList.length === 0) return;

    const history = JSON.parse(localStorage.getItem('lotto_history') || '[]');
    const newEntry = {
        id: Date.now(),
        date: new Date().toISOString(),
        model: currentModel,
        numbers: numbersList
    };

    history.unshift(newEntry);
    localStorage.setItem('lotto_history', JSON.stringify(history.slice(0, 1000))); // 최근 1000개만 유효
}

// 이력 로드 및 표시
function loadHistory() {
    const area = document.getElementById('historyArea');
    const history = JSON.parse(localStorage.getItem('lotto_history') || '[]');

    if (history.length === 0) {
        area.innerHTML = '<p class="text-gray-400">생성 이력이 없습니다.</p>';
        return;
    }

    area.innerHTML = history.map(entry => `
        <div class="bg-black/30 rounded-xl p-4 border border-white/5">
            <div class="flex justify-between items-start mb-3">
                <div class="text-xs text-gray-500">
                    📅 ${new Date(entry.date).toLocaleString('ko-KR')} | 
                    <span class="bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">${entry.model.toUpperCase()}</span>
                </div>
            </div>
            <div class="space-y-2">
                ${entry.numbers.map(nums => `
                    <div class="flex gap-1.5 flex-wrap">
                        ${nums.map(n => `<span class="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${getBallColor(n)}">${n}</span>`).join('')}
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

// 이력 삭제
function clearHistory() {
    if (confirm('모든 생성 이력을 삭제하시겠습니까?')) {
        localStorage.removeItem('lotto_history');
        loadHistory();
    }
}

// 공 색상 클래스
function getBallClass(num) {
    if (num <= 10) return 'ball-1-10';
    if (num <= 20) return 'ball-11-20';
    if (num <= 30) return 'ball-21-30';
    if (num <= 40) return 'ball-31-40';
    return 'ball-41-45';
}

// 공 색상 (폴백용)
function getBallColor(n) {
    if (n <= 10) return 'bg-yellow-500 text-black';
    if (n <= 20) return 'bg-blue-500 text-white';
    if (n <= 30) return 'bg-red-500 text-white';
    if (n <= 40) return 'bg-gray-600 text-white';
    return 'bg-green-500 text-white';
}
