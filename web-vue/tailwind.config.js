/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'void-navy': '#0f172a',
                'midnight-purple': '#2e1065',
                'neon-cyan': '#06b6d4',
                'vivid-pink': '#ec4899',
                'amber-gold': '#f59e0b',
                'primary': '#a855f7',
            },
            boxShadow: {
                'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
                'glow-pink': '0 0 15px rgba(236, 72, 153, 0.5)',
                'glow-cyan': '0 0 15px rgba(6, 182, 212, 0.5)',
                'glow-purple': '0 0 15px rgba(168, 85, 247, 0.5)',
            }
        },
    },
    plugins: [],
}
