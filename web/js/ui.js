// --- AI Lotto UI & Analytics Module V2 ---

// --- GLOBAL STATE ---
let globalDrawData = [];
let numberFrequency = {};
// --- SUPABASE CONFIG ---
const SB_URL = 'https://sfqlshdlqwqlkxdrfdke.supabase.co';
const SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNmcWxzaGRscXdxbGt4ZHJmZGtlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5MDM0NzUsImV4cCI6MjA4MTQ3OTQ3NX0.CMbJ_5IUxAifoNIzqdxu_3sz31AtOMw2vRBPxfxZzSk';
let supabase = null;

// --- INITIALIZATION ---
window.addEventListener('DOMContentLoaded', async () => {
    // Initialize Supabase
    try {
        if (window.supabase) {
            supabase = window.supabase.createClient(SB_URL, SB_KEY);
            console.log('Supabase client initialized');
        }
    } catch (e) {
        console.error('Supabase init failed:', e);
    }

    // Load default draws (Korea 645)
    await fetchAndBuildAnalytics('korea_645');
});

// --- ANALYTICS LOGIC ---

async function fetchAndBuildAnalytics(lotteryId) {
    try {
        const path = `data/${lotteryId}/draws.json`;
        const res = await fetch(path);
        if (!res.ok) return;

        const json = await res.json();
        const draws = json.draws || json;

        // Sort by date desc
        draws.sort((a, b) => new Date(b.date || b.drw_no_date) - new Date(a.date || a.drw_no_date));
        globalDrawData = draws;

        // Build Frequency Map (Last 50 draws)
        const recent = draws.slice(0, 50);
        numberFrequency = {};

        recent.forEach(draw => {
            const nums = [draw.drwt_no1, draw.drwt_no2, draw.drwt_no3, draw.drwt_no4, draw.drwt_no5, draw.drwt_no6, draw.bnus_no]
                .filter(n => n !== undefined && n !== null);

            // Fallback for array format
            if (nums.length === 0 && Array.isArray(draw.numbers)) {
                draw.numbers.forEach(n => nums.push(n));
            }

            nums.forEach(n => {
                numberFrequency[n] = (numberFrequency[n] || 0) + 1;
            });
        });

        // Initial Render
        renderHotNumbersChart();
        renderFrequencyMap();

    } catch (e) {
        console.warn('Analytics build failed:', e);
    }
}

// Render the Green Bar Chart (Hot Numbers)
function renderHotNumbersChart() {
    const container = document.getElementById('hotNumbersChart');
    if (!container) return;

    // Get Top 5 Hot Numbers
    const sorted = Object.entries(numberFrequency).sort((a, b) => b[1] - a[1]).slice(0, 5);
    if (sorted.length === 0) return;

    const maxFreq = sorted[0][1];

    // Simple Vertical Bar Chart
    container.innerHTML = sorted.map(([num, freq]) => {
        const height = Math.max(20, (freq / maxFreq) * 100);
        return `
        <div class="flex flex-col items-center gap-2 group">
            <div class="relative w-8 bg-white/5 rounded-t-lg overflow-hidden flex items-end transition-all group-hover:bg-white/10" style="height: 100px;">
                <div class="w-full bg-green-500 shadow-[0_0_15px_rgba(34,197,94,0.4)] animate-entry" style="height: ${height}%;"></div>
                <span class="absolute bottom-1 w-full text-center text-[10px] font-bold text-black/50">${freq}</span>
            </div>
            <span class="text-sm font-bold text-white">${num}</span>
        </div>
        `;
    }).join('');
}

// Render the Small Frequency Map
function renderFrequencyMap() {
    const container = document.getElementById('frequencyBarChart');
    if (!container) return;

    // Group into ranges
    const ranges = [0, 0, 0, 0, 0]; // 1-9, 10-19, 20-29, 30-39, 40-45
    Object.entries(numberFrequency).forEach(([num, freq]) => {
        const n = parseInt(num);
        if (n < 10) ranges[0] += freq;
        else if (n < 20) ranges[1] += freq;
        else if (n < 30) ranges[2] += freq;
        else if (n < 40) ranges[3] += freq;
        else ranges[4] += freq;
    });

    const maxRange = Math.max(...ranges);

    container.innerHTML = ranges.map((val, i) => {
        const h = maxRange > 0 ? (val / maxRange) * 100 : 0;
        const labels = ['1-9', '10s', '20s', '30s', '40s'];
        return `
        <div class="flex-1 flex flex-col justify-end group tooltip-container relative">
            <div class="w-full bg-purple-500/20 border-t border-purple-500/50 rounded-t-sm hover:bg-purple-500/40 transition-all" style="height: ${h}%"></div>
            <div class="absolute -top-8 left-1/2 -translate-x-1/2 bg-black text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap border border-white/10">
                ${labels[i]}: ${val}
            </div>
        </div>`;
    }).join('');
}

// Helper: Adjust Generation Count
window.adjustCount = function (delta) {
    const input = document.getElementById('countInput');
    if (input) {
        let val = parseInt(input.value) || 5;
        val += delta;
        if (val < 1) val = 1;
        if (val > 20) val = 20;
        input.value = val;
    }
}

// Helper: Calculate AI Confidence Score
function calculateConfidence(numbers) {
    let score = 50; // Base score

    // A. Sum Analysis
    const sum = numbers.reduce((a, b) => a + b, 0);
    if (sum >= 100 && sum <= 175) score += 15;
    else if (sum >= 80 && sum <= 200) score += 5;
    else score -= 10;

    // B. AC Value (Complexity)
    const diffs = new Set();
    for (let i = 0; i < numbers.length; i++) {
        for (let j = i + 1; j < numbers.length; j++) {
            diffs.add(Math.abs(numbers[i] - numbers[j]));
        }
    }
    const ac = diffs.size - (numbers.length - 1);
    if (ac >= 7) score += 15;
    else score -= 5;

    // C. Odd/Even Balance
    const odd = numbers.filter(n => n % 2 === 1).length;
    if (odd >= 2 && odd <= 4) score += 10;
    else score -= 5;

    // D. Heatmap Weight
    let hotCount = 0;
    numbers.forEach(n => {
        if ((numberFrequency[n] || 0) > 2) hotCount++;
    });
    score += (hotCount * 2);

    // E. Cap Score
    if (score > 98) score = 98;
    if (score < 40) score = 40 + Math.floor(Math.random() * 10);

    return score;
}

// --- HELPER: GET CURRENT LOTTERY ID ---
function getLotteryType() {
    // Check Desktop then Mobile
    const d = document.getElementById('lotterySelectDesktop');
    if (d && d.offsetParent !== null) return d.value; // Visible

    const m = document.getElementById('lotterySelectMobile');
    if (m && m.offsetParent !== null) return m.value; // Visible

    // Fallback to whichever exists
    return (d ? d.value : (m ? m.value : 'korea_645'));
}

// --- GLOBAL EVENT HANDLER: LOTTERY CHANGE ---
window.onLotteryChange = async function (val) {
    // Sync both selectors
    const d = document.getElementById('lotterySelectDesktop');
    const m = document.getElementById('lotterySelectMobile');

    if (d && d.value !== val) d.value = val;
    if (m && m.value !== val) m.value = val;

    console.log(`Lottery changed to: ${val}`);

    // Trigger Data Reload
    await fetchAndBuildAnalytics(val);
}

// --- CLOUD STORAGE ---

async function saveToSupabase(data) {
    if (!supabase) return;

    const payload = {
        created_at: new Date().toISOString(),
        numbers: data.numbers,
        model: data.model,
        lottery_type: getLotteryType(),
        user_id: localStorage.getItem('lotto_user_id') || (() => {
            const id = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('lotto_user_id', id);
            return id;
        })()
    };

    try {
        const { error } = await supabase.from('lotto_history').insert([payload]);
        if (error) {
            console.warn('Supabase save error:', error.message);
        } else {
            console.log('Saved to Supabase successfully');
        }
    } catch (err) {
        console.warn('Supabase network error:', err);
    }
}


// --- UI OVERRIDES (Interact with app.js) ---

// 1. Override displayResults to render into the new layout
window.displayResults = function (data) {
    // Hide Placeholder
    const placeholder = document.getElementById('resultsPlaceholder');
    if (placeholder) placeholder.classList.add('hidden');

    const area = document.getElementById('resultsArea');
    if (!area) return;

    if (!data.numbers || data.numbers.length === 0) {
        area.innerHTML = '<div class="p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-center">No numbers generated.</div>';
        return;
    }

    // Render Cards
    let html = '';
    data.numbers.forEach((set, i) => {
        const confidence = calculateConfidence(set.numbers);
        let ringColor = 'border-gray-600';
        let bgGlow = '';
        let scoreColor = 'text-gray-400';

        if (confidence >= 90) {
            ringColor = 'border-purple-500';
            bgGlow = 'shadow-[0_0_30px_-5px_rgba(168,85,247,0.3)] bg-purple-500/5';
            scoreColor = 'text-purple-400';
        } else if (confidence >= 80) {
            ringColor = 'border-blue-500';
            bgGlow = 'bg-blue-500/5';
            scoreColor = 'text-blue-400';
        }

        html += `
             <div class="glass-panel p-4 rounded-xl border border-white/5 flex items-center justify-between group hover:border-white/10 transition-all ${bgGlow} animation-entry" style="animation-delay: ${i * 0.1}s">
                <div class="flex items-center gap-4">
                    <div class="w-8 h-8 rounded-full border-2 ${ringColor} flex items-center justify-center text-xs font-bold ${scoreColor} bg-black/20">
                        ${confidence}
                    </div>
                    <div class="flex gap-2">
                        ${set.numbers.map(n => `
                            <span class="w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${getBallClass(n)} text-shadow-sm shadow-lg transform transition-transform group-hover:scale-110 text-white">
                                ${n}
                            </span>
                        `).join('')}
                    </div>
                </div>
                
                <div class="hidden md:flex flex-col items-end text-[10px] text-gray-500 gap-1">
                    <span class="font-mono">Sum: <b class="text-gray-300">${set.analysis.sum}</b></span>
                    <span class="font-mono">AC: <b class="text-gray-300">${set.analysis.ac_value}</b></span>
                </div>
             </div>
             `;
    });

    area.innerHTML = html;

    // Trigger Save
    const historyItem = {
        id: Date.now(),
        date: new Date().toISOString(),
        model: data.model,
        numbers: data.numbers
    };

    // Local Save (App Wrapper)
    if (window.saveHistoryEntry && window.saveHistoryEntry.name !== 'saveHistoryEntryWrapped') {
        window.saveHistoryEntry(historyItem);
    } else if (window.originalSaveHistoryEntry) {
        window.originalSaveHistoryEntry(historyItem);
    }

    // Cloud Save (Explicit)
    saveToSupabase(historyItem);

    // Refresh History Tab if visible
    if (document.getElementById('view-history') && !document.getElementById('view-history').classList.contains('hidden')) {
        setTimeout(loadHistory, 500);
    }
}

// 2. Override loadHistory (because layout might be different)
window.loadHistory = function () {
    const area = document.getElementById('historyArea');
    if (!area) return;

    const history = JSON.parse(localStorage.getItem('lotto_history') || '[]');

    if (history.length === 0) {
        area.innerHTML = '<div class="text-center py-10 opacity-50"><div class="text-4xl mb-2">📜</div><p>No generation history yet</p></div>';
        return;
    }

    area.innerHTML = history.reverse().map(entry => `
        <div class="glass-panel p-4 rounded-xl border-l-2 border-l-purple-500/50 hover:bg-white/5 transition-colors">
            <div class="flex justify-between items-start mb-3">
                <div class="flex items-center gap-2">
                     <span class="text-[10px] font-bold bg-purple-500/20 text-purple-200 px-2 py-0.5 rounded border border-purple-500/30">${entry.model ? entry.model.toUpperCase() : 'UNKNOWN'}</span>
                     <span class="text-xs text-gray-400">${new Date(entry.date || entry.generated_at || Date.now()).toLocaleString('ko-KR')}</span>
                </div>
            </div>
            <div class="space-y-2">
                ${entry.numbers ? (Array.isArray(entry.numbers) ? entry.numbers : []).map((nums, idx) => `
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] text-gray-600 font-mono w-4">#${idx + 1}</span>
                        <div class="flex gap-1.5 flex-wrap">
                            ${(Array.isArray(nums) ? nums : nums.numbers).map(n =>
        `<span class="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${getBallClass(n)} text-shadow shadow-lg text-white">${n}</span>`
    ).join('')}
                        </div>
                    </div>
                `).join('') : ''}
            </div>
        </div>
    `).join('');
}


// 3. Override selectModel to handle new Grid and Statistical Sections
window.selectModel = function (model) {
    if (window.appSelectModel) {
        window.appSelectModel(model); // Call core logic if available
    } else {
        // Fallback or explicit set
        if (typeof currentModel !== 'undefined') currentModel = model;
    }

    // Visual Update: Reset all cards
    const cards = document.querySelectorAll('.model-card');
    cards.forEach(c => {
        c.classList.remove('ring-2', 'ring-blue-500', 'ring-purple-500', 'ring-green-500', 'ring-pink-500', 'ring-cyan-500', 'bg-blue-500/10', 'bg-purple-500/10', 'bg-green-500/10', 'bg-pink-500/10', 'bg-cyan-500/10');
        const dot = c.querySelector('.active-dot');
        if (dot) dot.classList.add('hidden');
    });

    // Identify target IDs for both sections
    let targetIds = [];
    let ringClass = 'ring-blue-500';
    let bgClass = 'bg-blue-500/10';

    if (model === 'transformer') {
        targetIds = ['card-transformer-v3', 'card-transformer-stat'];
        ringClass = 'ring-blue-500';
        bgClass = 'bg-blue-500/10';
    } else if (model === 'lstm') {
        targetIds = ['card-lstm-v3', 'card-lstm-stat'];
        ringClass = 'ring-purple-500';
        bgClass = 'bg-purple-500/10';
    } else if (model === 'vector') {
        targetIds = ['card-physics-v3', 'card-physics-stat', 'card-cold-stat'];
        ringClass = 'ring-green-500';
        bgClass = 'bg-green-500/10';
    }

    // Apply highlighting to all matches
    targetIds.forEach(id => {
        const activeCard = document.getElementById(id);
        if (activeCard) {
            let currentRing = ringClass;
            let currentBg = bgClass;

            // Specific highlight for Cold Theory
            if (id === 'card-cold-stat') {
                currentRing = 'ring-cyan-500';
                currentBg = 'bg-cyan-500/10';
            }

            activeCard.classList.add('ring-2', currentRing, currentBg);
            const dot = activeCard.querySelector('.active-dot');
            if (dot) dot.classList.remove('hidden');
        }
    });
}

