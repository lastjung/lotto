/**
 * AI Lotto Analyzer - Analysis & Charts Logic (Unified)
 * Handles both Dashboard charts and History Analysis tab charts.
 */

window.analysisCharts = {};
window.analysisLoaded = false;
window.dashboardDrawData = [];

/**
 * Fetch draws data and build all analytics (Dashboard + History)
 */
async function fetchAndBuildAllAnalytics(lotteryId) {
    try {
        const config = window.lotteryConfigs[lotteryId] || { ball_count: 6 };
        let draws = [];

        // 1. Try Supabase first
        if (window.supabaseClient) {
            try {
                const { data, error } = await window.supabaseClient
                    .from("lotto_draws")
                    .select("*")
                    .eq("lottery_type", lotteryId)
                    .order("draw_date", { ascending: false })
                    .limit(100); // Fetch more for better history

                if (!error && data && data.length > 0) {
                    console.log(`✅ [Analysis] Supabase loaded: ${data.length} draws`);
                    draws = data.map((d) => ({
                        draw_no: d.draw_number,
                        draw_date: d.draw_date,
                        numbers: d.numbers,
                        bonus: d.bonus_number,
                    }));
                }
            } catch (e) {
                console.warn("⚠️ Supabase fetch failed, using JSON fallback:", e);
            }
        }

        // 2. Fallback to JSON if Supabase failed
        if (draws.length === 0) {
            console.log("📁 [Analysis] Using JSON fallback");
            let res = await fetch(`/data/${lotteryId}/draws.json`);
            if (!res.ok) res = await fetch(`../data/${lotteryId}/draws.json`);
            if (res.ok) {
                const json = await res.json();
                draws = json.draws || json;
            }
        }

        if (draws.length === 0) throw new Error("No data found");

        window.dashboardDrawData = draws;

        // Perform all analysis renderings
        updateDashboardCharts(draws, config.ball_count);
        updateHistoryTabCharts(draws, config.ball_count, lotteryId);

        window.analysisLoaded = true;
    } catch (e) {
        console.error("❌ Analysis failed:", e);
    }
}

/** --- DASHBOARD CHARTS --- **/

function updateDashboardCharts(draws, ballCount) {
    const recent = draws.slice(0, 52);
    
    // 1. Hot Numbers (Dashboard Bar Chart)
    const freqMap = {};
    recent.forEach(d => {
        d.numbers?.forEach(n => { freqMap[n] = (freqMap[n] || 0) + 1; });
    });
    
    const container = document.getElementById('hotNumbersChart');
    if (container) {
        const sorted = Object.entries(freqMap).sort((a, b) => b[1] - a[1]).slice(0, ballCount);
        if (sorted.length > 0) {
            const maxFreq = sorted[0][1];
            container.innerHTML = sorted.map(([num, freq]) => {
                const height = Math.max(20, (freq / maxFreq) * 100);
                return `
                <div class="flex flex-col items-center gap-2 group">
                    <div class="relative w-8 bg-white/5 rounded-t-lg overflow-hidden flex items-end transition-all group-hover:bg-white/10" style="height: 100px;">
                        <div class="w-full bg-green-500 shadow-[0_0_15px_rgba(34,197,94,0.4)]" style="height: ${height}%;"></div>
                        <span class="absolute bottom-1 w-full text-center text-[10px] font-bold text-black/50">${freq}</span>
                    </div>
                    <span class="text-sm font-bold text-white">${num}</span>
                </div>`;
            }).join('');
        }
    }

    // 2. Sum Distribution (Dashboard Header Chart)
    const allSums = draws.map(d => d.numbers?.reduce((a, b) => a + b, 0) || 0);
    renderDashboardSumChart(allSums);
}

function renderDashboardSumChart(allSums) {
    const canvas = document.getElementById('sumDistributionChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const binSize = 10;
    const startBin = Math.floor(Math.min(...allSums) / binSize) * binSize;
    const endBin = Math.ceil(Math.max(...allSums) / binSize) * binSize;
    const labels = [], data = [];
    for (let i = startBin; i < endBin; i += binSize) {
        labels.push(`${i}-${i + binSize - 1}`);
        data.push(allSums.filter(s => s >= i && s < i + binSize).length);
    }

    if (window.analysisCharts.dashboardSum) window.analysisCharts.dashboardSum.destroy();
    window.analysisCharts.dashboardSum = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data,
                borderColor: '#a855f7',
                backgroundColor: 'rgba(168, 85, 247, 0.2)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { display: false }, x: { display: false } }
        }
    });
}

/** --- HISTORY TAB ANALYSIS --- **/

function updateHistoryTabCharts(draws, ballCount, lotteryId) {
    const oneYearDraws = draws.slice(0, 52);
    
    // Update Stats Badge
    const drawCountEl = document.getElementById("drawCount");
    if (drawCountEl) drawCountEl.innerText = oneYearDraws.length;

    if (oneYearDraws.length > 0) {
        const start = oneYearDraws[oneYearDraws.length - 1].draw_date || "Unknown";
        const end = oneYearDraws[0].draw_date || "Unknown";
        const periodRangeEl = document.getElementById("periodRange");
        if (periodRangeEl) periodRangeEl.innerText = `${start} ~ ${end}`;
    }

    // Frequency Analysis
    const freq = new Array(80).fill(0);
    const bonusFreq = new Array(80).fill(0);
    let maxNum = 0, maxBonus = 0;

    oneYearDraws.forEach((d) => {
        d.numbers?.forEach((n) => { freq[n]++; if (n > maxNum) maxNum = n; });
        if (d.bonus !== undefined) { bonusFreq[d.bonus]++; if (d.bonus > maxBonus) maxBonus = d.bonus; }
    });

    const ranked = [];
    for (let i = 1; i <= maxNum; i++) ranked.push({ num: i, count: freq[i] });
    ranked.sort((a, b) => b.count - a.count);

    renderAnalysisBalls("hotNumbers", ranked.slice(0, ballCount));
    renderAnalysisBalls("coldNumbers", ranked.slice(-ballCount).reverse());

    if (maxBonus > 0) {
        const bRanked = [];
        for (let i = 1; i <= maxBonus; i++) bRanked.push({ num: i, count: bonusFreq[i] });
        bRanked.sort((a, b) => b.count - a.count);
        renderAnalysisBalls("hotBonus", bRanked.slice(0, 3));
        renderAnalysisBalls("coldBonus", bRanked.slice(-3).reverse());
    }

    // Render Tab Charts
    renderAnalysisFreqChart(ranked.slice(0, 20));
    
    const oddData = { odd: 0, even: 0 };
    oneYearDraws.forEach(d => d.numbers?.forEach(n => (n % 2 === 0 ? oddData.even++ : oddData.odd++)));
    renderAnalysisOddEvenChart(oddData.odd, oddData.even);

    const midPoint = maxNum / 2;
    const hiLoData = { low: 0, high: 0 };
    oneYearDraws.forEach(d => d.numbers?.forEach(n => (n <= midPoint ? hiLoData.low++ : hiLoData.high++)));
    renderAnalysisLowHighChart(hiLoData.low, hiLoData.high);

    const last30 = draws.slice(0, 30);
    const labels = last30.map(d => d.draw_no || "").reverse();
    const sums = last30.map(d => d.numbers?.reduce((a, b) => a + b, 0) || 0).reverse();
    renderAnalysisSumChart(labels, sums);

    const acValues = last30.map(d => Utils.calculateAC(d.numbers)).reverse();
    renderAnalysisACChart(labels, acValues);

    const allSums = draws.map(d => d.numbers?.reduce((a, b) => a + b, 0) || 0);
    renderAnalysisSumDistChart(allSums);

    const endFreq = new Array(10).fill(0);
    draws.forEach(d => d.numbers?.forEach(n => endFreq[n % 10]++));
    renderAnalysisEndDigitChart(endFreq);
}

function renderAnalysisBalls(id, list) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = list.map((item) => `
        <div class="flex flex-col items-center gap-1">
            <div class="lotto-ball-v2 ${Utils.getBallClass(item.num)} text-sm shadow-md">${item.num}</div>
            <span class="text-xs font-bold text-gray-400">${item.count}회</span>
        </div>`).join("");
}

// ... (Sub-chart rendering functions follow, same as before but using window.analysisCharts) ...

function renderAnalysisFreqChart(data) {
    const ctx = document.getElementById("freqChart")?.getContext("2d");
    if (!ctx) return;
    if (window.analysisCharts.freq) window.analysisCharts.freq.destroy();
    window.analysisCharts.freq = new Chart(ctx, {
        type: "bar",
        data: {
            labels: data.map((d) => d.num),
            datasets: [{
                label: "출현 횟수",
                data: data.map((d) => d.count),
                backgroundColor: "#6366f1",
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true }, x: { grid: { display: false } } },
        },
    });
}

function renderAnalysisOddEvenChart(odd, even) {
    const ctx = document.getElementById("oddEvenChart")?.getContext("2d");
    if (!ctx) return;
    if (window.analysisCharts.oddEven) window.analysisCharts.oddEven.destroy();
    const total = odd + even;
    window.analysisCharts.oddEven = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Odd (홀)", "Even (짝)"],
            datasets: [{ data: [odd, even], backgroundColor: ["#818cf8", "#34d399"], borderWidth: 0 }],
        },
        plugins: [ChartDataLabels],
        options: {
            cutout: "55%",
            rotation: 180,
            plugins: {
                legend: { position: "bottom", labels: { color: "#9ca3af" } },
                datalabels: {
                    color: "#fff",
                    font: { weight: "bold", size: 14 },
                    formatter: (value) => ((value / total) * 100).toFixed(0) + "%",
                },
            },
        },
    });
}

function renderAnalysisLowHighChart(low, high) {
    const ctx = document.getElementById("lowHighChart")?.getContext("2d");
    if (!ctx) return;
    if (window.analysisCharts.lowHigh) window.analysisCharts.lowHigh.destroy();
    const total = low + high;
    window.analysisCharts.lowHigh = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["High (고)", "Low (저)"],
            datasets: [{ data: [high, low], backgroundColor: ["#f87171", "#60a5fa"], borderWidth: 0 }],
        },
        plugins: [ChartDataLabels],
        options: {
            cutout: "55%",
            rotation: -90,
            plugins: {
                legend: { position: "bottom", labels: { color: "#9ca3af" } },
                datalabels: {
                    color: "#fff",
                    font: { weight: "bold", size: 14 },
                    formatter: (value) => ((value / total) * 100).toFixed(0) + "%",
                },
            },
        },
    });
}

function renderAnalysisSumChart(labels, data) {
    const ctx = document.getElementById("sumChart")?.getContext("2d");
    if (!ctx) return;
    if (window.analysisCharts.sum) window.analysisCharts.sum.destroy();
    window.analysisCharts.sum = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "합계",
                data,
                borderColor: "#34d399",
                backgroundColor: "rgba(52, 211, 153, 0.1)",
                fill: true,
                tension: 0.4,
                pointRadius: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { display: false } },
        },
    });
}

function renderAnalysisACChart(labels, data) {
    const ctx = document.getElementById("acChart")?.getContext("2d");
    if (!ctx) return;
    if (window.analysisCharts.ac) window.analysisCharts.ac.destroy();
    window.analysisCharts.ac = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "AC Value",
                data,
                borderColor: "#ec4899",
                backgroundColor: "rgba(236, 72, 153, 0.1)",
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { suggestedMin: 0, suggestedMax: 10 }, x: { display: false } },
        },
    });
}

function renderAnalysisSumDistChart(allSums) {
    const ctx = document.getElementById("sumDistChart")?.getContext("2d");
    if (!ctx || allSums.length === 0) return;
    if (window.analysisCharts.sumDist) window.analysisCharts.sumDist.destroy();

    const binSize = 10;
    const minSum = Math.floor(Math.min(...allSums) / binSize) * binSize;
    const maxSum = Math.ceil(Math.max(...allSums) / binSize) * binSize;
    const labels = [], distData = [];
    for (let i = minSum; i < maxSum; i += binSize) {
        labels.push(`${i}~${i + binSize - 1}`);
        distData.push(allSums.filter((s) => s >= i && s < i + binSize).length);
    }

    window.analysisCharts.sumDist = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{ label: "빈도수", data: distData, backgroundColor: "#a78bfa", borderRadius: 4 }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true }, x: { grid: { display: false } } },
        },
    });
}

function renderAnalysisEndDigitChart(data) {
    const ctx = document.getElementById("endDigitChart")?.getContext("2d");
    if (!ctx) return;
    if (window.analysisCharts.endDigit) window.analysisCharts.endDigit.destroy();
    window.analysisCharts.endDigit = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            datasets: [{ label: "빈도", data, backgroundColor: "#60a5fa", borderRadius: 4 }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true }, x: { grid: { display: false } } },
        },
    });
}

// Global hook for tab changes or lottery changes
window.refreshAnalysis = function(lotteryId) {
    fetchAndBuildAllAnalytics(lotteryId || window.currentLotteryId || 'korea_645');
};

/**
 * Switch between "Latest Draws" and "Generated History" in History Tab
 */
window.switchHistoryTab = function(tab) {
    const drawsTab = document.getElementById("history-tab-draws");
    const generatedTab = document.getElementById("history-tab-generated");
    const drawsBtn = document.getElementById("history-tab-btn-draws");
    const generatedBtn = document.getElementById("history-tab-btn-generated");

    if (!drawsTab || !generatedTab || !drawsBtn || !generatedBtn) return;

    if (tab === "draws") {
        drawsTab.classList.remove("hidden");
        generatedTab.classList.add("hidden");
        drawsBtn.classList.add("bg-blue-600/20", "text-blue-400", "border-b-2", "border-blue-500", "font-bold");
        drawsBtn.classList.remove("text-gray-400", "font-medium");
        generatedBtn.classList.remove("bg-blue-600/20", "text-blue-400", "border-b-2", "border-blue-500", "font-bold");
        generatedBtn.classList.add("text-gray-400", "font-medium");

        if (!window.analysisLoaded) {
            fetchAndBuildAllAnalytics(window.currentLotteryId || 'korea_645');
        }
    } else {
        drawsTab.classList.add("hidden");
        generatedTab.classList.remove("hidden");
        generatedBtn.classList.add("bg-blue-600/20", "text-blue-400", "border-b-2", "border-blue-500", "font-bold");
        generatedBtn.classList.remove("text-gray-400", "font-medium");
        drawsBtn.classList.remove("bg-blue-600/20", "text-blue-400", "border-b-2", "border-blue-500", "font-bold");
        drawsBtn.classList.add("text-gray-400", "font-medium");
    }
};

// Auto-run on load
document.addEventListener('DOMContentLoaded', () => {
    const lid = window.currentLotteryId || 'korea_645';
    fetchAndBuildAllAnalytics(lid);
});
