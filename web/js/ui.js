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
        console.warn('Supabase init failed (UI):', e);
    }
});

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

// --- CLOUD STORAGE ---
async function saveToSupabase(data) {
    if (!supabaseClient) return;

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
