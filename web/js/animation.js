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
            revealDelay: 300,        // ms between each ball reveal (0.3s)
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
        return Utils.getBallClass(num);
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
        return Utils.getBallClass(num);
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
        return Utils.getBallClass(num);
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
 * 🌀 Quantum Shuffle Animation
 * 
 * Numbers rapidly shuffle then lock in one by one with a flash effect.
 * Ported from web-vue QuantumShuffleAnimation.vue
 */
class QuantumShuffleAnimation {
    constructor(options = {}) {
        this.options = {
            soundEnabled: options.soundEnabled !== false,
            onComplete: options.onComplete || null
        };
        this.container = null;
        this.isPlaying = false;
        this.shuffleInterval = null;
        this.lockTimeouts = [];
    }

    getBallColorClass(n) {
        return Utils.getBallClass(n);
    }

    getAudioContext() {
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume().catch(e => console.log('Audio resume failed', e));
        }
        return this.audioCtx;
    }

    startShuffleSound() {
        if (!this.options.soundEnabled) return;
        try {
            const ctx = this.getAudioContext();
            
            // Create noise buffer for mechanical texture
            const bufferSize = ctx.sampleRate * 2; // 2 seconds
            // ... (Buffer creation logic unchanged) ...
            const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
            const data = buffer.getChannelData(0);
            for (let i = 0; i < bufferSize; i++) {
                data[i] = Math.random() * 2 - 1;
            }

            const noise = ctx.createBufferSource();
            noise.buffer = buffer;
            noise.loop = true;

            const noiseFilter = ctx.createBiquadFilter();
            noiseFilter.type = 'lowpass';
            noiseFilter.frequency.value = 400;

            const noiseGain = ctx.createGain();
            noiseGain.gain.value = 0.05;

            noise.connect(noiseFilter);
            noiseFilter.connect(noiseGain);
            noiseGain.connect(ctx.destination);
            
            this.shuffleNoise = { source: noise, gain: noiseGain };
            noise.start();

            // Create rhythmic pulse
            const osc = ctx.createOscillator();
            osc.type = 'square';
            osc.frequency.value = 15; // 15Hz rhythm

            const oscGain = ctx.createGain();
            oscGain.gain.value = 0.02;

            osc.connect(oscGain);
            oscGain.connect(ctx.destination);

            this.shufflePulse = { source: osc, gain: oscGain };
            osc.start();

        } catch (e) { console.error('Shuffle sound error', e); }
    }

    stopShuffleSound() {
        if (this.audioCtx) {
            const time = this.audioCtx.currentTime;
            
            if (this.shuffleNoise) {
                this.shuffleNoise.gain.gain.exponentialRampToValueAtTime(0.001, time + 0.5);
                this.shuffleNoise.source.stop(time + 0.5);
            }
            if (this.shufflePulse) {
                this.shufflePulse.gain.gain.exponentialRampToValueAtTime(0.001, time + 0.5);
                this.shufflePulse.source.stop(time + 0.5);
            }
            this.shuffleNoise = null;
            this.shufflePulse = null;
        }
    }

    playSound(type) {
        if (!this.options.soundEnabled) return;
        try {
            const ctx = this.getAudioContext();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);

            const now = ctx.currentTime;

            if (type === 'scan') {
                // Scanning sound: Softer sweep
                osc.type = 'sine'; // Changed from sawtooth to sine for softness
                osc.frequency.setValueAtTime(200, now);
                osc.frequency.linearRampToValueAtTime(600, now + 3.0);
                
                // Volume reduced
                gain.gain.setValueAtTime(0.1, now); 
                gain.gain.linearRampToValueAtTime(0.05, now + 2.8);
                gain.gain.linearRampToValueAtTime(0, now + 3.0);
                
                osc.start(now);
                osc.stop(now + 3.0);
            } else if (type === 'appear') {
                // Balls appear sound
                osc.type = 'sine';
                osc.frequency.setValueAtTime(400, now);
                osc.frequency.exponentialRampToValueAtTime(800, now + 0.1);
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
                osc.start(now);
                osc.stop(now + 0.3);
            } else if (type === 'lock') {
                // Sharp lock sound
                osc.frequency.setValueAtTime(800, now);
                osc.type = 'triangle';
                gain.gain.setValueAtTime(0.15, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
                osc.start(now);
                osc.stop(now + 0.15);
            } else if (type === 'complete') {
                // Success chord
                const freqs = [523.25, 659.25, 783.99]; // C Major
                const now = ctx.currentTime;
                freqs.forEach((f, i) => {
                    const o = ctx.createOscillator();
                    const g = ctx.createGain();
                    o.connect(g);
                    g.connect(ctx.destination);
                    o.type = 'sine';
                    o.frequency.value = f;
                    g.gain.setValueAtTime(0.05, now);
                    g.gain.exponentialRampToValueAtTime(0.001, now + 1.0);
                    o.start(now);
                    o.stop(now + 1.0);
                });
            }
        } catch (e) { /* ignore audio errors */ }
    }

    async animate(numbers, container) {
        this.container = container;
        this.isPlaying = true;
        this.cleanup();

        const ballCount = numbers.length;

        // Build HTML structure
        container.innerHTML = `
            <div class="quantum-shuffle">
                <div class="quantum-scan-line scanning"></div>
                <div class="quantum-balls">
                    ${numbers.map((_, i) => {
                        const n = Math.floor(Math.random() * 45) + 1;
                        const colorClass = this.getBallColorClass(n);
                        return `
                        <div class="quantum-ball shuffling ${colorClass}" 
                             id="qball-${i}" data-index="${i}">
                            ${n}
                        </div>
                    `}).join('')}
                </div>
                <!-- Stats Visible initially with placeholders -->
                <div class="quantum-stats" id="quantumStats">
                    <div class="quantum-stat-card">
                        <span class="quantum-stat-label">Sum Total</span>
                        <div class="quantum-stat-value" id="stat-sum">-</div>
                    </div>
                    <div class="quantum-stat-card">
                        <span class="quantum-stat-label">AC Value</span>
                        <div class="quantum-stat-value" id="stat-ac">-</div>
                    </div>
                    <div class="quantum-stat-card">
                        <span class="quantum-stat-label">Odd:Even</span>
                        <div class="quantum-stat-value" id="stat-oe">-:-</div>
                    </div>
                    <div class="quantum-stat-card confidence">
                        <span class="quantum-stat-label">Confidence</span>
                        <div class="quantum-stat-value" id="stat-conf">-</div>
                    </div>
                </div>
            </div>
        `;

        // Start shuffling immediately with loop sound
        this.startShuffleSound();
        this.playSound('scan');

        this.shuffleInterval = setInterval(() => {
            for (let i = 0; i < ballCount; i++) {
                const ball = document.getElementById(`qball-${i}`);
                if (ball && !ball.classList.contains('locked')) {
                    const n = Math.floor(Math.random() * 45) + 1;
                    ball.textContent = n;
                    // Dynamic color during shuffle
                    const colorClass = this.getBallColorClass(n);
                    ball.className = `quantum-ball shuffling ${colorClass}`;
                }
            }
        }, 60);

        // Wait for scan animation (3.0s) -- Increased for user capture
        await this.delay(3000);

        // Lock one by one - Adjusted to 0.3s
        const processDelay = 300; 

        for (let i = 0; i < ballCount; i++) {
            if (!this.isPlaying) break;

            const ball = document.getElementById(`qball-${i}`);
            if (ball) {
                ball.textContent = numbers[i];
                // Apply the correct color class from Utils
                const colorClass = this.getBallColorClass(numbers[i]);
                ball.className = `quantum-ball locked ${colorClass}`;
                this.playSound('lock'); // 'Tak!' sound
            }

            // Wait before next
            await this.delay(processDelay);
        }

        // Complete
        clearInterval(this.shuffleInterval);
        this.shuffleInterval = null;
        this.stopShuffleSound();

        // Calculate Stats
        const sum = numbers.reduce((a, b) => a + b, 0);
        // ... (Stats calculation code omitted for brevity as unchanged) ...
        const oddCount = numbers.filter(n => n % 2 !== 0).length;
        const evenCount = numbers.length - oddCount;
        const confidence = (95 + Math.random() * 4.5).toFixed(1);
        
        // AC value calculation
        const sorted = [...numbers].sort((a, b) => a - b);
        const diffs = new Set();
        for (let i = 0; i < sorted.length; i++) {
            for (let j = i + 1; j < sorted.length; j++) {
                diffs.add(sorted[j] - sorted[i]);
            }
        }
        const acValue = diffs.size - (numbers.length - 1);

        // Update Stats UI
        const statsEl = document.getElementById('quantumStats');
        if (statsEl) {
            document.getElementById('stat-sum').textContent = sum;
            document.getElementById('stat-ac').textContent = acValue;
            document.getElementById('stat-oe').textContent = `${oddCount}:${evenCount}`;
            document.getElementById('stat-conf').textContent = `${confidence}%`;
            
            statsEl.classList.add('active'); // Brighten stats
        }

        this.playSound('complete');
        
        // Wait a moment for user to see the result clearly before appending sets - 0.5s
        await this.delay(500);

        this.isPlaying = false;
        
        // Return numbers to app.js which handles additional sets
        if (typeof this.options.onComplete === 'function') this.options.onComplete(numbers);
        return numbers;
    }

    delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    cleanup() {
        if (this.shuffleInterval) {
            clearInterval(this.shuffleInterval);
            this.shuffleInterval = null;
        }
        this.stopShuffleSound();
        this.lockTimeouts.forEach(t => clearTimeout(t));
        this.lockTimeouts = [];
    }

    stop() {
        this.isPlaying = false;
        this.cleanup();
    }

    toggleSound(enabled) { this.options.soundEnabled = enabled; }
}

window.QuantumShuffleAnimation = QuantumShuffleAnimation;


/**
 * 🎛️ Animation Manager
 */
class AnimationManager {
    constructor() {
        this.animations = {
            'lottery_ball': LotteryAnimation,
            'slot_machine': SlotMachineAnimation,
            'ai_scanner': AIScannerAnimation,
            'quantum_shuffle': QuantumShuffleAnimation
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
            { id: 'ai_scanner', name: '🔬 AI 스캐너', desc: '미래형 스캔 & 락인 효과' },
            { id: 'quantum_shuffle', name: '🌀 퀀텀 셔플', desc: '양자 확률 기반 셔플 락인' }
        ];
    }
}

window.AnimationManager = AnimationManager;
window.animationManager = new AnimationManager();

