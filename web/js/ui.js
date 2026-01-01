// --- AI Lotto UI & Analytics Module V2 ---

// --- GLOBAL STATE ---
let globalDrawData = [];
let numberFrequency = {};
// --- SUPABASE CONFIG ---
// const SB_URL = '...'; // [FIX] Defined in app.js
// const SB_KEY = '...'; // [FIX] Defined in app.js
// let supabase = null;  // [FIX] Defined in app.js

// --- INITIALIZATION ---
window.addEventListener('DOMContentLoaded', async () => {
    // Initialize Supabase
    try {
        if (window.supabase) {
            // Use global supabaseClient from app.js
            if (!supabaseClient && typeof SB_URL !== 'undefined') {
                supabaseClient = window.supabase.createClient(SB_URL, SB_KEY);
            }
            console.log('Supabase client initialized (UI)');
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
        const { error } = await supabaseClient.from('lotto_history').insert([payload]);
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
/*
 * [DISABLED] displayResults override
 * Using app.js version which includes lottery type and model header.
 * The ui.js version was missing the header info.
 */
// window.displayResults = function (data) { ... }


/*
 * [DISABLED] loadHistory and selectModel overrides
 * These are now handled by app.js to avoid function collision.
 * Uncomment below if you need to restore ui.js-specific behavior.
 */

// // 2. Override loadHistory (because layout might be different)
// window.loadHistory = function () { ... }

// // 3. Override selectModel to handle new Grid and Statistical Sections
// window.selectModel = function (model, isInit = false) { ... }
