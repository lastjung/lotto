/**
 * AI Lotto Analyzer - Shared Utilities
 * Consolidates redundant logic across app.js, animation.js, and index.html
 */

const Utils = {
    /**
     * Calculate AC (Complexity) value of a number set
     */
    calculateAC(numbers) {
        if (!numbers || numbers.length < 2) return 0;
        const sorted = [...numbers].sort((a, b) => a - b);
        const diffs = new Set();

        for (let i = 0; i < sorted.length; i++) {
            for (let j = i + 1; j < sorted.length; j++) {
                diffs.add(sorted[j] - sorted[i]);
            }
        }
        return diffs.size - (numbers.length - 1);
    },

    /**
     * Get CSS class for lotto ball color based on number
     */
    getBallClass(num) {
        if (num <= 10) return 'ball-1-10';
        if (num <= 20) return 'ball-11-20';
        if (num <= 30) return 'ball-21-30';
        if (num <= 40) return 'ball-31-40';
        return 'ball-41-45';
    },

    /**
     * Get Legacy/Fallback color classes (for charts or older views)
     */
    getBallColorLegacy(n) {
        if (n <= 10) return 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20';
        if (n <= 20) return 'bg-blue-500 text-white shadow-lg shadow-blue-500/20';
        if (n <= 30) return 'bg-red-500 text-white shadow-lg shadow-red-500/20';
        if (n <= 40) return 'bg-gray-600 text-white shadow-lg shadow-gray-500/20';
        return 'bg-green-500 text-white shadow-lg shadow-green-500/20';
    },

    /**
     * Check for consecutive numbers
     */
    hasConsecutive(numbers, minCount = 3) {
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
    },

    /**
     * Get the display label of the currently selected lottery
     */
    getSelectedLotteryLabel() {
        const sel = document.getElementById('lotterySelectDesktop') || 
                    document.getElementById('lotterySelectMobile') || 
                    document.getElementById('lotterySelect');
        return sel && sel.options[sel.selectedIndex] ? 
               sel.options[sel.selectedIndex].text.trim() : 
               'Korea Lotto 6/45';
    },

    /**
     * Format date to local string
     */
    formatDate(dateStr) {
        return new Date(dateStr).toLocaleString();
    }
};

// Lottery configurations - Shared across app and analysis
window.lotteryConfigs = {
    korea_645: { ball_count: 6, ball_range: [1, 45], name: "Korea Lotto 6/45" },
    usa_powerball: { ball_count: 5, ball_range: [1, 69], name: "USA Powerball" },
    usa_megamillions: { ball_count: 5, ball_range: [1, 70], name: "USA Mega Millions" },
    canada_649: { ball_count: 6, ball_range: [1, 49], name: "Canada 6/49" },
    japan_loto6: { ball_count: 6, ball_range: [1, 43], name: "Japan Loto 6" }
};

// Dynamic load from config/lotteries.json
(async function loadLotteryConfigs() {
    try {
        let res = await fetch("/config/lotteries.json");
        if (!res.ok) res = await fetch("config/lotteries.json");
        if (!res.ok) res = await fetch("../config/lotteries.json");

        if (res.ok) {
            const json = await res.json();
            window.lotteryConfigs = json;
            console.log("✅ Lottery configs loaded from config/lotteries.json");
        }
    } catch (e) {
        console.warn("⚠️ Using fallback lotteryConfigs:", e);
    }
})();

// Expose to window for global access
window.Utils = Utils;
