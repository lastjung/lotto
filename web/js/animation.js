/**
 * 🎱 Lottery Ball Animation Module
 * 
 * A modular animation system for lottery number generation.
 * Features: Ball mixing, pop-out animation, sound effects
 * 
 * @author AI Lotto Analyzer
 * @version 1.0.0
 */

class LotteryAnimation {
    constructor(options = {}) {
        this.container = null;
        this.options = {
            ballCount: 45,           // Total balls in the machine
            revealDelay: 400,        // ms between each ball reveal
            mixDuration: 2000,       // ms for mixing animation
            soundEnabled: true,
            onComplete: null,        // Callback when animation completes
            ...options
        };

        // Sound effects
        this.sounds = {
            mix: null,
            pop: null,
            complete: null
        };

        this.isPlaying = false;
        this.init();
    }

    init() {
        this.createSounds();
    }

    createSounds() {
        if (!this.options.soundEnabled) return;

        // Create audio context for generating sounds
        try {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn('Web Audio API not supported');
            this.options.soundEnabled = false;
        }
    }

    // Generate a "pop" sound effect
    playPopSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;

        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        osc.frequency.setValueAtTime(800, this.audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(300, this.audioCtx.currentTime + 0.1);

        gain.gain.setValueAtTime(0.3, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.1);

        osc.start(this.audioCtx.currentTime);
        osc.stop(this.audioCtx.currentTime + 0.1);
    }

    // Generate a "complete" celebration sound
    playCompleteSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;

        const notes = [523, 659, 784, 1047]; // C5, E5, G5, C6

        notes.forEach((freq, i) => {
            setTimeout(() => {
                const osc = this.audioCtx.createOscillator();
                const gain = this.audioCtx.createGain();

                osc.connect(gain);
                gain.connect(this.audioCtx.destination);

                osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
                osc.type = 'sine';

                gain.gain.setValueAtTime(0.2, this.audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.3);

                osc.start(this.audioCtx.currentTime);
                osc.stop(this.audioCtx.currentTime + 0.3);
            }, i * 100);
        });
    }

    // Generate mixing/rumble sound
    playMixSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;

        const bufferSize = this.audioCtx.sampleRate * 2;
        const buffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
        const data = buffer.getChannelData(0);

        for (let i = 0; i < bufferSize; i++) {
            data[i] = (Math.random() * 2 - 1) * 0.1;
        }

        const noise = this.audioCtx.createBufferSource();
        const filter = this.audioCtx.createBiquadFilter();
        const gain = this.audioCtx.createGain();

        noise.buffer = buffer;
        filter.type = 'lowpass';
        filter.frequency.value = 200;

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.audioCtx.destination);

        gain.gain.setValueAtTime(0.3, this.audioCtx.currentTime);
        gain.gain.linearRampToValueAtTime(0, this.audioCtx.currentTime + 2);

        noise.start();
        noise.stop(this.audioCtx.currentTime + 2);
    }

    // Get ball color class based on number
    getBallColorClass(num) {
        if (num <= 10) return 'ball-yellow';
        if (num <= 20) return 'ball-blue';
        if (num <= 30) return 'ball-red';
        if (num <= 40) return 'ball-gray';
        return 'ball-green';
    }

    // Create the animation container HTML
    createAnimationHTML() {
        return `
            <div class="lotto-machine">
                <div class="machine-dome">
                    <div class="mixing-balls" id="mixingBalls">
                        ${Array(12).fill(0).map((_, i) =>
            `<div class="mixing-ball ball-${['yellow', 'blue', 'red', 'gray', 'green'][i % 5]}" 
                                  style="--delay: ${i * 0.1}s; --x: ${Math.random() * 100}%; --y: ${Math.random() * 100}%"></div>`
        ).join('')}
                    </div>
                </div>
                <div class="machine-chute">
                    <div class="chute-opening"></div>
                </div>
                <div class="revealed-balls" id="revealedBalls"></div>
            </div>
        `;
    }

    // Main animation method
    async animate(numbers, container) {
        if (this.isPlaying) return;
        this.isPlaying = true;

        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        if (!this.container) {
            console.error('Animation container not found');
            this.isPlaying = false;
            return;
        }

        // Clear and setup
        this.container.innerHTML = this.createAnimationHTML();
        this.container.classList.add('animation-active');

        const mixingBalls = this.container.querySelector('#mixingBalls');
        const revealedBalls = this.container.querySelector('#revealedBalls');

        // Resume audio context if suspended
        if (this.audioCtx?.state === 'suspended') {
            await this.audioCtx.resume();
        }

        // Phase 1: Mixing animation
        mixingBalls.classList.add('mixing');
        this.playMixSound();

        await this.delay(this.options.mixDuration);

        // Phase 2: Reveal balls one by one
        for (let i = 0; i < numbers.length; i++) {
            const num = numbers[i];

            // Create ball element
            const ball = document.createElement('div');
            ball.className = `revealed-ball ${this.getBallColorClass(num)} pop-in`;
            ball.innerHTML = `<span>${num}</span>`;
            ball.style.animationDelay = '0s';

            // Add to revealed area
            revealedBalls.appendChild(ball);

            // Play pop sound
            this.playPopSound();

            // Wait before next ball
            await this.delay(this.options.revealDelay);
        }

        // Phase 3: Complete
        mixingBalls.classList.remove('mixing');
        this.playCompleteSound();

        // Add celebration effect
        this.container.classList.add('complete');

        await this.delay(500);

        this.isPlaying = false;

        // Callback
        if (typeof this.options.onComplete === 'function') {
            this.options.onComplete(numbers);
        }

        return numbers;
    }

    // Utility: delay promise
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Stop animation
    stop() {
        this.isPlaying = false;
        if (this.container) {
            this.container.classList.remove('animation-active', 'complete');
        }
    }

    // Toggle sound
    toggleSound(enabled) {
        this.options.soundEnabled = enabled;
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LotteryAnimation;
}

// Also expose globally for script tag usage
window.LotteryAnimation = LotteryAnimation;


/**
 * 🎰 Slot Machine Animation
 * 
 * Casino-style spinning reels animation
 */
class SlotMachineAnimation {
    constructor(options = {}) {
        this.container = null;
        this.options = {
            spinDuration: 2000,      // ms for spinning
            reelDelay: 200,          // ms delay between reels stopping
            soundEnabled: true,
            onComplete: null,
            ...options
        };

        this.isPlaying = false;
        this.audioCtx = null;
        this.init();
    }

    init() {
        if (this.options.soundEnabled) {
            try {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                console.warn('Web Audio API not supported');
                this.options.soundEnabled = false;
            }
        }
    }

    // Spinning sound effect
    playSpinSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;

        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        osc.frequency.setValueAtTime(150, this.audioCtx.currentTime);
        osc.type = 'sawtooth';

        gain.gain.setValueAtTime(0.1, this.audioCtx.currentTime);
        gain.gain.linearRampToValueAtTime(0, this.audioCtx.currentTime + 2);

        osc.start(this.audioCtx.currentTime);
        osc.stop(this.audioCtx.currentTime + 2);
    }

    // Stop/click sound
    playStopSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;

        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        osc.frequency.setValueAtTime(400, this.audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(200, this.audioCtx.currentTime + 0.1);

        gain.gain.setValueAtTime(0.3, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.15);

        osc.start(this.audioCtx.currentTime);
        osc.stop(this.audioCtx.currentTime + 0.15);
    }

    // Win celebration sound
    playWinSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;

        const notes = [392, 523, 659, 784]; // G4, C5, E5, G5
        notes.forEach((freq, i) => {
            setTimeout(() => {
                const osc = this.audioCtx.createOscillator();
                const gain = this.audioCtx.createGain();

                osc.connect(gain);
                gain.connect(this.audioCtx.destination);

                osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
                osc.type = 'square';

                gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.2);

                osc.start(this.audioCtx.currentTime);
                osc.stop(this.audioCtx.currentTime + 0.2);
            }, i * 80);
        });
    }

    // Get ball color class
    getBallColorClass(num) {
        if (num <= 10) return 'ball-yellow';
        if (num <= 20) return 'ball-blue';
        if (num <= 30) return 'ball-red';
        if (num <= 40) return 'ball-gray';
        return 'ball-green';
    }

    // Create slot machine HTML
    createSlotHTML(numbers) {
        const reels = numbers.map((num, i) => `
            <div class="slot-reel" data-index="${i}">
                <div class="reel-container">
                    <div class="reel-strip" id="reel-${i}">
                        ${this.createReelNumbers(num)}
                    </div>
                </div>
                <div class="reel-frame"></div>
            </div>
        `).join('');

        return `
            <div class="slot-machine">
                <div class="slot-header">
                    <span class="slot-title">🎰 LUCKY DRAW 🎰</span>
                </div>
                <div class="slot-reels">
                    ${reels}
                </div>
                <div class="slot-lever">
                    <div class="lever-ball"></div>
                    <div class="lever-stick"></div>
                </div>
            </div>
        `;
    }

    // Create number strip for a reel
    createReelNumbers(finalNum) {
        // Create array of random numbers ending with the final number
        const randomNums = Array(20).fill(0).map(() => Math.floor(Math.random() * 45) + 1);
        randomNums.push(finalNum); // Final number at end

        return randomNums.map(n => `
            <div class="reel-number ${this.getBallColorClass(n)}">
                <span>${n}</span>
            </div>
        `).join('');
    }

    // Main animation
    async animate(numbers, container) {
        if (this.isPlaying) return;
        this.isPlaying = true;

        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        if (!this.container) {
            console.error('Animation container not found');
            this.isPlaying = false;
            return;
        }

        // Handle empty numbers array
        if (!numbers || numbers.length === 0) {
            this.container.innerHTML = `
                <div class="slot-machine">
                    <div class="slot-header">
                        <span class="slot-title">🎰 LUCKY DRAW 🎰</span>
                    </div>
                    <div class="slot-reels" style="padding: 2rem; text-align: center;">
                        <div style="color: #f97316; font-size: 1.2rem;">
                            ⚠️ 생성된 번호가 없습니다
                        </div>
                        <div style="color: #888; margin-top: 0.5rem; font-size: 0.9rem;">
                            필터 설정을 확인하거나 다시 시도해주세요
                        </div>
                    </div>
                </div>
            `;
            this.container.classList.add('slot-active');
            this.isPlaying = false;
            if (typeof this.options.onComplete === 'function') {
                this.options.onComplete([]);
            }
            return;
        }

        // Resume audio context
        if (this.audioCtx?.state === 'suspended') {
            await this.audioCtx.resume();
        }

        // Setup HTML
        this.container.innerHTML = this.createSlotHTML(numbers);
        this.container.classList.add('slot-active');

        // Start spinning
        this.playSpinSound();

        // Animate each reel
        for (let i = 0; i < numbers.length; i++) {
            const strip = this.container.querySelector(`#reel-${i}`);
            if (strip) {
                strip.classList.add('spinning');
            }
        }

        // Stop reels one by one
        await this.delay(this.options.spinDuration);

        for (let i = 0; i < numbers.length; i++) {
            const strip = this.container.querySelector(`#reel-${i}`);
            if (strip) {
                strip.classList.remove('spinning');
                strip.classList.add('stopped');
                this.playStopSound();
            }
            await this.delay(this.options.reelDelay);
        }

        // Complete
        this.playWinSound();
        this.container.classList.add('slot-complete');

        await this.delay(500);

        this.isPlaying = false;

        if (typeof this.options.onComplete === 'function') {
            this.options.onComplete(numbers);
        }

        return numbers;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    stop() {
        this.isPlaying = false;
        if (this.container) {
            this.container.classList.remove('slot-active', 'slot-complete');
        }
    }

    toggleSound(enabled) {
        this.options.soundEnabled = enabled;
    }
}

window.SlotMachineAnimation = SlotMachineAnimation;


/**
 * 🔬 AI Scanner Animation
 * 
 * Futuristic scanning effect with number lock-in
 */
class AIScannerAnimation {
    constructor(options = {}) {
        this.container = null;
        this.options = {
            scanDuration: 1500,
            lockDelay: 300,
            soundEnabled: true,
            onComplete: null,
            ...options
        };

        this.isPlaying = false;
        this.audioCtx = null;
        this.init();
    }

    init() {
        if (this.options.soundEnabled) {
            try {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                this.options.soundEnabled = false;
            }
        }
    }

    playScanSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.frequency.setValueAtTime(1200, this.audioCtx.currentTime);
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.1, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.1);
        osc.start(this.audioCtx.currentTime);
        osc.stop(this.audioCtx.currentTime + 0.1);
    }

    playLockSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.frequency.setValueAtTime(880, this.audioCtx.currentTime);
        osc.frequency.setValueAtTime(1320, this.audioCtx.currentTime + 0.05);
        osc.type = 'square';
        gain.gain.setValueAtTime(0.2, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.15);
        osc.start(this.audioCtx.currentTime);
        osc.stop(this.audioCtx.currentTime + 0.15);
    }

    playCompleteSound() {
        if (!this.options.soundEnabled || !this.audioCtx) return;
        [523, 659, 784, 1047].forEach((freq, i) => {
            setTimeout(() => {
                const osc = this.audioCtx.createOscillator();
                const gain = this.audioCtx.createGain();
                osc.connect(gain);
                gain.connect(this.audioCtx.destination);
                osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
                osc.type = 'sine';
                gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.2);
                osc.start(this.audioCtx.currentTime);
                osc.stop(this.audioCtx.currentTime + 0.2);
            }, i * 80);
        });
    }

    getBallColorClass(num) {
        if (num <= 10) return 'ball-yellow';
        if (num <= 20) return 'ball-blue';
        if (num <= 30) return 'ball-red';
        if (num <= 40) return 'ball-gray';
        return 'ball-green';
    }

    createScannerHTML(numbers) {
        const slots = numbers.map((num, i) => `
            <div class="scanner-slot" data-index="${i}" data-final="${num}">
                <div class="scanner-display"><span class="scanning-number">--</span></div>
                <div class="scanner-glow"></div>
            </div>
        `).join('');

        return `
            <div class="ai-scanner">
                <div class="scanner-header">
                    <span class="scanner-title">🔬 AI ANALYSIS</span>
                    <span class="scanner-status">SCANNING...</span>
                </div>
                <div class="scanner-slots">${slots}</div>
                <div class="scanner-progress"><div class="progress-bar"></div></div>
            </div>
        `;
    }

    async animate(numbers, container) {
        if (this.isPlaying) return;
        this.isPlaying = true;

        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) { this.isPlaying = false; return; }

        // Handle empty numbers array
        if (!numbers || numbers.length === 0) {
            this.container.innerHTML = `
                <div class="ai-scanner">
                    <div class="scanner-header">
                        <span class="scanner-title">🔬 AI ANALYSIS</span>
                        <span class="scanner-status" style="color: #f97316;">ERROR</span>
                    </div>
                    <div class="scanner-slots" style="padding: 2rem; text-align: center;">
                        <div style="color: #f97316; font-size: 1.2rem;">
                            ⚠️ 생성된 번호가 없습니다
                        </div>
                        <div style="color: #888; margin-top: 0.5rem; font-size: 0.9rem;">
                            필터 설정을 확인하거나 다시 시도해주세요
                        </div>
                    </div>
                </div>
            `;
            this.container.classList.add('scanner-active');
            this.isPlaying = false;
            if (typeof this.options.onComplete === 'function') {
                this.options.onComplete([]);
            }
            return;
        }

        if (this.audioCtx?.state === 'suspended') await this.audioCtx.resume();

        this.container.innerHTML = this.createScannerHTML(numbers);
        this.container.classList.add('scanner-active');

        const slots = this.container.querySelectorAll('.scanner-slot');
        const progressBar = this.container.querySelector('.progress-bar');
        const statusEl = this.container.querySelector('.scanner-status');

        // Scanning phase
        const scanInterval = setInterval(() => {
            slots.forEach(slot => {
                if (!slot.classList.contains('locked')) {
                    slot.querySelector('.scanning-number').textContent =
                        (Math.floor(Math.random() * 45) + 1).toString().padStart(2, '0');
                }
            });
            this.playScanSound();
        }, 80);

        progressBar.style.transition = `width ${this.options.scanDuration}ms linear`;
        progressBar.style.width = '100%';
        await this.delay(this.options.scanDuration);
        clearInterval(scanInterval);

        // Lock phase
        statusEl.textContent = 'LOCKING...';
        for (let i = 0; i < numbers.length; i++) {
            const slot = slots[i];
            slot.classList.add('locked');
            slot.querySelector('.scanning-number').textContent = numbers[i].toString().padStart(2, '0');
            slot.querySelector('.scanner-display').classList.add(this.getBallColorClass(numbers[i]));
            this.playLockSound();
            await this.delay(this.options.lockDelay);
        }

        statusEl.textContent = 'COMPLETE!';
        this.container.classList.add('scanner-complete');
        this.playCompleteSound();
        await this.delay(500);

        this.isPlaying = false;
        if (typeof this.options.onComplete === 'function') this.options.onComplete(numbers);
        return numbers;
    }

    delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
    stop() { this.isPlaying = false; if (this.container) this.container.classList.remove('scanner-active', 'scanner-complete'); }
    toggleSound(enabled) { this.options.soundEnabled = enabled; }
}

window.AIScannerAnimation = AIScannerAnimation;


/**
 * 🎛️ Animation Manager
 */
class AnimationManager {
    constructor() {
        this.animations = {
            'lottery_ball': LotteryAnimation,
            'slot_machine': SlotMachineAnimation,
            'ai_scanner': AIScannerAnimation
        };

        this.currentType = localStorage.getItem('animationType') || 'lottery_ball';
        this.currentInstance = null;
        this.soundEnabled = localStorage.getItem('animationSound') !== 'false';
    }

    getAnimation(onComplete) {
        const AnimClass = this.animations[this.currentType] || LotteryAnimation;
        this.currentInstance = new AnimClass({
            soundEnabled: this.soundEnabled,
            onComplete: onComplete
        });
        return this.currentInstance;
    }

    setType(type) {
        if (this.animations[type]) {
            this.currentType = type;
            localStorage.setItem('animationType', type);
        }
    }

    getType() { return this.currentType; }

    toggleSound(enabled) {
        this.soundEnabled = enabled;
        localStorage.setItem('animationSound', enabled);
        if (this.currentInstance) this.currentInstance.toggleSound(enabled);
    }

    getAvailableTypes() {
        return [
            { id: 'lottery_ball', name: '🎱 로또볼 추첨기', desc: '공이 튀며 나오는 실제 추첨 효과' },
            { id: 'slot_machine', name: '🎰 슬롯머신', desc: '카지노 스타일 릴 회전' },
            { id: 'ai_scanner', name: '🔬 AI 스캐너', desc: '미래형 스캔 & 락인 효과' }
        ];
    }
}

window.AnimationManager = AnimationManager;
window.animationManager = new AnimationManager();

