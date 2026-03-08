/**
 * Powered by:  onnx-community/kitten-tts-nano-0.1-ONNX
 *              phonemizer@1.2.1 (CDN)
 *              onnxruntime-web  (auto-loaded — no HTML import needed)
 * Config options (all optional):
 *   voiceUrl     {string}  URL to a .bin voice-embedding file  (default: './default.bin')
 *   speed        {number}  Playback speed multiplier           (default: 1.0)
 *   debug        {bool}    Verbose console logging             (default: false)
 *   concurrency  {number}  Max parallel generation jobs        (default: 2)
 *   segmentMax   {number}  Char limit before a chunk is split  (default: 220)
 *
 * Usage:
 *   const tts = new KittenTTS({ voiceUrl: './default.bin', debug: true });
 *   await tts.ready;             // wait for model + voice to load
 *   tts.speak('Hello world!');   // call from a user-gesture handler
 */

const _KITTENTTS_SAMPLE_RATE   = 24000;
const _KITTENTTS_CACHE_NAME    = 'kitten-tts-v1';
const _KITTENTTS_BASE_URL      = 'https://huggingface.co/onnx-community/kitten-tts-nano-0.1-ONNX/resolve/main';
const _KITTENTTS_MODEL_URL     = `${_KITTENTTS_BASE_URL}/onnx/model_quantized.onnx`;
const _KITTENTTS_TOKENIZER_URL = `${_KITTENTTS_BASE_URL}/tokenizer.json`;
const _KITTENTTS_ORT_VERSION   = '1.24.3';
const _KITTENTTS_ORT_CDN       = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${_KITTENTTS_ORT_VERSION}/dist/`;
const _KITTENTTS_ORT_ESM       = `${_KITTENTTS_ORT_CDN}ort.min.mjs`;

// ---------------------------------------------------------------------------
//  Inner Engine
// ---------------------------------------------------------------------------
class _KittenEngine {
    constructor(debug) {
        this.session        = null;
        this.vocab          = {};
        this.voiceEmbedding = null;
        this.debug          = debug;
        this._ort           = null;
        // ONNX WASM does not support concurrent session.run() calls.
        // Serialize all inference through a promise chain acting as a mutex.
        this._inferenceQueue = Promise.resolve();
    }

    log(msg) {
        if (this.debug) console.log(`%c[KittenEngine] ${msg}`, 'color:#10b981;font-weight:bold;');
    }

    async _loadOrt() {
        if (typeof ort !== 'undefined') {
            this.log('Using globally available ort');
            this._ort = ort;
            return;
        }

        this.log(`Fetching onnxruntime-web@${_KITTENTTS_ORT_VERSION} from CDN…`);

        try {
            const mod  = await import(_KITTENTTS_ORT_ESM);
            this._ort  = mod.default ?? mod;
            this._ort.env.wasm.wasmPaths = _KITTENTTS_ORT_CDN;
            this.log('ort loaded via ESM import');
            return;
        } catch (e) {
            this.log(`ESM import failed (${e.message}), trying <script> fallback…`);
        }

        await new Promise((resolve, reject) => {
            const script   = document.createElement('script');
            script.src     = `${_KITTENTTS_ORT_CDN}ort.min.js`;
            script.onload  = resolve;
            script.onerror = () => reject(new Error('Failed to load ort via <script> tag'));
            document.head.appendChild(script);
        });

        if (typeof ort === 'undefined') throw new Error('onnxruntime-web loaded but window.ort is still undefined');

        this._ort = ort;
        this._ort.env.wasm.wasmPaths = _KITTENTTS_ORT_CDN;
        this.log('ort loaded via <script> tag');
    }

    async _cachedFetch(url) {
        if (!('caches' in window)) {
            this.log(`CacheStorage unavailable — fetching direct: ${url}`);
            return fetch(url);
        }
        const cache  = await caches.open(_KITTENTTS_CACHE_NAME);
        const cached = await cache.match(url);
        if (cached) { this.log(`Cache HIT: ${url}`); return cached; }

        this.log(`Cache MISS — downloading: ${url}`);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
        await cache.put(url, response.clone());
        return response;
    }

    async load(onStatus) {
        try {
            onStatus('Loading ONNX Runtime…');
            await this._loadOrt();

            onStatus('Loading model…');
            const modelBuf = await (await this._cachedFetch(_KITTENTTS_MODEL_URL)).arrayBuffer();
            this.session   = await this._ort.InferenceSession.create(modelBuf, {
                executionProviders: ['wasm'],
            });
            this.log('ONNX session created');

            onStatus('Loading tokenizer…');
            const tokData = await (await this._cachedFetch(_KITTENTTS_TOKENIZER_URL)).json();
            this.vocab    = tokData.model.vocab;
            this.log('Tokenizer ready');

            return true;
        } catch (err) {
            onStatus('Load error: ' + err.message);
            console.error('[KittenEngine] load failed', err);
            return false;
        }
    }

    async clearModelCache() {
        if (!('caches' in window)) { this.log('CacheStorage unavailable'); return false; }
        const deleted = await caches.delete(_KITTENTTS_CACHE_NAME);
        this.log(deleted ? '✅ Model cache cleared' : 'Cache not found');
        return deleted;
    }

    async loadVoice(url) {
        this.log(`Loading voice: ${url}`);
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching voice: ${url}`);
        const buf = await resp.arrayBuffer();
        if (buf.byteLength === 0) throw new Error(`Voice file is empty: ${url}`);
        this.voiceEmbedding = new Float32Array(buf);
        this.log(`Voice loaded — ${this.voiceEmbedding.length} floats`);
    }

    async tokenize(text) {
        let phonemes = '';
        try {
            const { phonemize } = await import('https://cdn.jsdelivr.net/npm/phonemizer@1.2.1/dist/phonemizer.js');
            const arr = await phonemize(text, 'en-us');
            phonemes  = arr.join(' ').replace(/\s+/g, ' ').trim();
        } catch {
            this.log('Phonemizer unavailable — ASCII fallback');
            phonemes = text.toLowerCase().replace(/[^a-z\s.,!?;:]/g, '');
        }

        const str    = `$${phonemes}$`;
        const tokens = [];
        for (const ch of str) {
            const id = this.vocab[ch];
            if (id !== undefined) tokens.push(BigInt(id));
        }
        this.log(`Tokenized "${text.slice(0, 40)}" → ${tokens.length} tokens`);
        return new BigInt64Array(tokens);
    }

    async generate(text, speed = 1.0) {
        if (!this.session)        throw new Error('Engine not loaded');
        if (!this.voiceEmbedding) throw new Error('No voice embedding loaded');

        const tokens = await this.tokenize(text);

        // Chain onto the inference queue so only one session.run() is active at a time.
        // Each call waits for the previous to finish before starting.
        const waveform = await (this._inferenceQueue = this._inferenceQueue.then(async () => {
            this.log(`Inference start: "${text.slice(0, 40)}"`);
            const results = await this.session.run({
                input_ids : new this._ort.Tensor('int64',   tokens,                    [1, tokens.length]),
                style     : new this._ort.Tensor('float32', this.voiceEmbedding,       [1, 256]),
                speed     : new this._ort.Tensor('float32', new Float32Array([speed]), [1]),
            });

            const wav = results.waveform.data;
            let max = 0;
            for (const v of wav) max = Math.max(max, Math.abs(v));
            if (max > 0) for (let i = 0; i < wav.length; i++) wav[i] *= (0.95 / max);
            this.log(`Generated ${wav.length} samples`);
            return wav;
        }));

        return waveform;
    }
}

// ---------------------------------------------------------------------------
//  Public KittenTTS class
// ---------------------------------------------------------------------------
export class KittenTTS {
    constructor(config = {}) {
        this.config = {
            voiceUrl    : config.voiceUrl    ?? './default.bin',
            speed       : config.speed       ?? 1.0,
            debug       : config.debug       ?? false,
            concurrency : config.concurrency ?? 2,
            segmentMax  : config.segmentMax  ?? 220,
        };

        this._engine    = new _KittenEngine(this.config.debug);
        this._audioCtx  = null; // created lazily on first speak() to satisfy autoplay policy
        this.isReady    = false;

        this._textQueue     = [];
        this._pendingAudio  = new Map();
        this._activeJobs    = 0;
        this._nextGenIndex  = 0;
        this._nextPlayIndex = 0;
        this._isPlaying     = false;
        this._interrupted   = false;

        // Expose a promise callers can await
        this.ready = this._init();
    }

    log(msg, data = '') {
        if (this.config.debug)
            console.log(`%c[KittenTTS] ${msg}`, 'color:#2563eb;font-weight:bold;', data);
    }

    // -----------------------------------------------------------------------
    //  Init — runs in the background; await tts.ready to know when done
    // -----------------------------------------------------------------------
    async _init() {
        this.log('Initializing…');

        const ok = await this._engine.load(msg => this.log(msg));
        if (!ok) throw new Error('KittenTTS engine failed to load');

        if (this.config.voiceUrl) {
            try {
                await this._engine.loadVoice(this.config.voiceUrl);
            } catch (e) {
                console.warn('[KittenTTS] ⚠️ Voice load failed:', e.message);
            }
        }

        this.isReady = true;
        this.log('✅ Ready');
    }

    // -----------------------------------------------------------------------
    //  Create/resume the AudioContext.
    //  MUST be called inside a user-gesture handler to satisfy autoplay policy.
    // -----------------------------------------------------------------------
    async _ensureAudioContext() {
        if (!this._audioCtx) {
            this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: _KITTENTTS_SAMPLE_RATE,
            });
            this.log(`AudioContext created (state: ${this._audioCtx.state})`);
        }

        if (this._audioCtx.state === 'suspended') {
            this.log('AudioContext suspended — resuming…');
            await this._audioCtx.resume();
        }

        if (this._audioCtx.state !== 'running') {
            throw new Error(
                `AudioContext is "${this._audioCtx.state}". ` +
                'speak() must be called from a user-gesture handler (click, keydown, etc.).'
            );
        }
    }

    // -----------------------------------------------------------------------
    //  Public API
    // -----------------------------------------------------------------------

    /**
     * Speak rawText.
     * Returns a Promise that resolves once all segments have been *queued*
     * (not necessarily finished playing).
     * MUST be called from a user-gesture handler.
     */
    async speak(rawText) {
        if (!this.isReady) {
            this.log('Not ready yet — awaiting init…');
            await this.ready;
        }

        await this._ensureAudioContext();

        this._interrupted = false;

        const segments = this._parseText(rawText);
        this.log(`Queuing ${segments.length} segment(s)`);

        for (const text of segments) {
            this._textQueue.push({ text, genIndex: this._nextGenIndex++ });
        }

        this._pumpGenerators();
    }

    clearModelCache() { return this._engine.clearModelCache(); }

    interrupt() {
        this.log('🛑 Interrupt');
        this._interrupted   = true;
        this._textQueue     = [];
        this._pendingAudio  = new Map();
        this._activeJobs    = 0;
        this._nextGenIndex  = 0;
        this._nextPlayIndex = 0;

        if (this._audioCtx) {
            this._audioCtx.suspend().then(() => {
                this._audioCtx.resume();
                this._isPlaying = false;
            });
        }
    }

    // -----------------------------------------------------------------------
    //  Internal: generation pump
    // -----------------------------------------------------------------------

    _pumpGenerators() {
        while (
            this._activeJobs < this.config.concurrency &&
            this._textQueue.length > 0
        ) {
            const job = this._textQueue.shift();
            this._runJob(job);
        }
    }

    async _runJob({ text, genIndex }) {
        if (this._interrupted) return;

        this._activeJobs++;
        this.log(`Generating [${genIndex}] "${text.slice(0, 40)}…"`);

        try {
            const waveform = await this._engine.generate(text, this.config.speed);
            if (this._interrupted) { this._activeJobs--; return; }

            const buffer = this._waveformToBuffer(waveform);
            this.log(`Segment [${genIndex}] ready (${buffer.duration.toFixed(2)}s)`);

            this._pendingAudio.set(genIndex, buffer);
            this._playNextInOrder();

        } catch (err) {
            console.error(`[KittenTTS] ❌ Generation error [${genIndex}]:`, err);
        }

        this._activeJobs--;
        this._pumpGenerators();
    }

    // -----------------------------------------------------------------------
    //  Internal: ordered playback
    // -----------------------------------------------------------------------

    _playNextInOrder() {
        if (this._isPlaying) return;
        if (!this._pendingAudio.has(this._nextPlayIndex)) return;

        this._isPlaying = true;
        const buffer    = this._pendingAudio.get(this._nextPlayIndex);
        this._pendingAudio.delete(this._nextPlayIndex);
        this._nextPlayIndex++;

        const source  = this._audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this._audioCtx.destination);
        this.log(`▶ Playing segment (${buffer.duration.toFixed(2)}s)`);

        source.onended = () => {
            this._isPlaying = false;
            setTimeout(() => this._playNextInOrder(), 250);
        };

        source.start();
    }

    // -----------------------------------------------------------------------
    //  Internal: helpers
    // -----------------------------------------------------------------------

    _waveformToBuffer(waveform) {
        const buf = this._audioCtx.createBuffer(1, waveform.length, _KITTENTTS_SAMPLE_RATE);
        buf.copyToChannel(new Float32Array(waveform), 0);
        return buf;
    }

    _parseText(rawText) {
        let text = rawText
            .replace(/\*\*([^*]+)\*\*/g, '$1')
            .replace(/\*([^*]+)\*/g,     '$1')
            .replace(/`([^`]+)`/g,       '$1')
            .replace(/#{1,6}\s+/g,       '')
            .replace(/\s+/g,             ' ')
            .trim();

        const ABBREVS     = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'St.', 'vs.', 'etc.'];
        const PLACEHOLDER = '\x00DOT\x00';
        for (const abbr of ABBREVS)
            text = text.split(abbr).join(abbr.replace('.', PLACEHOLDER));
        text = text.replace(/(\d+)\.(\d)/g, `$1${PLACEHOLDER}$2`);

        const raw = text.match(/[^.!?]+[.!?]+["'\])]?\s*|[^.!?]+$/g) ?? [text];

        const segments = [];
        for (const chunk of raw) {
            const s = chunk.replace(new RegExp(PLACEHOLDER, 'g'), '.').trim();
            if (!s) continue;

            if (s.length <= this.config.segmentMax) {
                segments.push(s);
            } else {
                const parts = s.split(/(?<=[,;—])\s+/).map(p => p.trim()).filter(Boolean);
                let acc = '';
                for (const part of parts) {
                    if (acc.length + part.length + 1 > this.config.segmentMax) {
                        if (acc) segments.push(acc);
                        acc = part;
                    } else {
                        acc = acc ? `${acc} ${part}` : part;
                    }
                }
                if (acc) segments.push(acc);
            }
        }

        // Merge segments under 3 words into the next segment if they fit.
        const merged = [];
        let carry = '';
        for (let i = 0; i < segments.length; i++) {
            const seg       = carry ? `${carry} ${segments[i]}` : segments[i];
            const wordCount = seg.trim().split(/\s+/).length;
            const isShort   = wordCount < 3;
            const hasNext   = i + 1 < segments.length;

            if (isShort && hasNext && (seg.length + 1 + segments[i + 1].length) <= this.config.segmentMax) {
                // Too short — carry it forward and merge with the next iteration
                carry = seg;
            } else {
                merged.push(seg);
                carry = '';
            }
        }
        // If the last segment was left in carry (no next to merge into), flush it
        if (carry) merged.push(carry);

        return merged.filter(s => s.length > 0);
    }
}

// ---------------------------------------------------------------------------
//  Convenience globals
// ---------------------------------------------------------------------------
window.initTTS = (config) => {
    window.tts       = new KittenTTS(config);
    window.speak     = (text) => window.tts.speak(text);
    window.interrupt = ()     => window.tts.interrupt();
    return window.tts.ready;  // so you can: await initTTS({...})
};
