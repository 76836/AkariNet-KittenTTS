/**
 * Powered by:  onnx-community/kitten-tts-nano-0.1-ONNX
 *              phonemizer@1.2.1 (CDN)
 *              onnxruntime-web  (must be loaded by the host page)
 * Config options (all optional):
 *   voiceUrl     {string}  URL to a .bin voice-embedding file  (default: './default.bin')
 *   speed        {number}  Playback speed multiplier           (default: 1.0)
 *   debug        {bool}    Verbose console logging             (default: false)
 *   concurrency  {number}  Max parallel generation jobs        (default: 2)
 *   segmentMax   {number}  Char limit before a chunk is split  (default: 220)
 */

const _KITTENTTS_SAMPLE_RATE = 24000;
const _KITTENTTS_CACHE_NAME  = 'kitten-tts-v1';
const _KITTENTTS_BASE_URL    = 'https://huggingface.co/onnx-community/kitten-tts-nano-0.1-ONNX/resolve/main';
const _KITTENTTS_MODEL_URL   = `${_KITTENTTS_BASE_URL}/onnx/model_quantized.onnx`;
const _KITTENTTS_TOKENIZER_URL = `${_KITTENTTS_BASE_URL}/tokenizer.json`;

// ---------------------------------------------------------------------------
//  Inner Engine — wraps the raw KittenTTS ONNX session
// ---------------------------------------------------------------------------
class _KittenEngine {
    constructor(debug) {
        this.session        = null;
        this.vocab          = {};
        this.voiceEmbedding = null;
        this.debug          = debug;

        if (typeof ort !== 'undefined') {
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
        }
    }

    log(msg) {
        if (this.debug) console.log(`%c[KittenEngine] ${msg}`, 'color:#10b981;font-weight:bold;');
    }

    /**
     * Fetch a URL, reading from CacheStorage first.
     * On a cache miss the response is fetched from the network and stored.
     */
    async _cachedFetch(url) {
        // CacheStorage isn't available in all contexts (e.g. non-secure origins)
        if (!('caches' in window)) {
            this.log(`CacheStorage unavailable — fetching direct: ${url}`);
            return fetch(url);
        }

        const cache    = await caches.open(_KITTENTTS_CACHE_NAME);
        const cached   = await cache.match(url);

        if (cached) {
            this.log(`Cache HIT: ${url}`);
            return cached;
        }

        this.log(`Cache MISS — downloading: ${url}`);
        const response = await fetch(url);

        if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);

        // Clone before consuming — cache keeps one copy, we return the other
        await cache.put(url, response.clone());
        return response;
    }

    /** Load the ONNX model + tokenizer. Calls onStatus(string) for progress. */
    async load(onStatus) {
        try {
            onStatus('Loading model…');
            const modelBuf = await (await this._cachedFetch(_KITTENTTS_MODEL_URL)).arrayBuffer();
            this.session   = await ort.InferenceSession.create(modelBuf, { executionProviders: ['wasm'] });
            this.log('ONNX session created');

            onStatus('Loading tokenizer…');
            const tokData = await (await this._cachedFetch(_KITTENTTS_TOKENIZER_URL)).json();
            this.vocab     = tokData.model.vocab;
            this.log('Tokenizer ready');

            return true;
        } catch (err) {
            onStatus('Load error: ' + err.message);
            console.error('[KittenEngine] load failed', err);
            return false;
        }
    }

    /** Delete the CacheStorage bucket, forcing a fresh download next time. */
    async clearModelCache() {
        if (!('caches' in window)) {
            this.log('CacheStorage unavailable — nothing to clear');
            return false;
        }
        const deleted = await caches.delete(_KITTENTTS_CACHE_NAME);
        this.log(deleted ? '✅ Model cache cleared' : 'Cache not found (already clear)');
        return deleted;
    }

    /** Load a .bin voice-embedding file (raw Float32, 256 values). */
    async loadVoice(url) {
        try {
            this.log(`Loading voice: ${url}`);
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
            this.voiceEmbedding = new Float32Array(await resp.arrayBuffer());
            this.log(`Voice loaded (${this.voiceEmbedding.length} floats)`);
        } catch (err) {
            console.error('[KittenEngine] voice load failed', err);
            throw err;
        }
    }

    /** Convert text → phoneme tokens using CDN phonemizer (falls back to ASCII). */
    async tokenize(text) {
        let phonemes = '';
        try {
            const { phonemize } = await import('https://cdn.jsdelivr.net/npm/phonemizer@1.2.1/dist/phonemizer.js');
            const arr = await phonemize(text, 'en-us');
            phonemes  = arr.join(' ').replace(/\s+/g, ' ').trim();
        } catch {
            this.log('Phonemizer unavailable — using ASCII fallback');
            phonemes = text.toLowerCase().replace(/[^a-z\s.,!?;:]/g, '');
        }

        const str    = `$${phonemes}$`;
        const tokens = [];
        for (const ch of str) {
            const id = this.vocab[ch];
            if (id !== undefined) tokens.push(BigInt(id));
        }
        return new BigInt64Array(tokens);
    }

    /** Generate raw waveform Float32Array for one text segment. */
    async generate(text, speed = 1.0) {
        if (!this.session)        throw new Error('Engine not loaded');
        if (!this.voiceEmbedding) throw new Error('No voice embedding loaded');

        const tokens  = await this.tokenize(text);
        const results = await this.session.run({
            input_ids : new ort.Tensor('int64',   tokens,                  [1, tokens.length]),
            style     : new ort.Tensor('float32', this.voiceEmbedding,     [1, 256]),
            speed     : new ort.Tensor('float32', new Float32Array([speed]), [1]),
        });

        // Normalize to ±0.95
        const waveform = results.waveform.data;
        let max = 0;
        for (const v of waveform) max = Math.max(max, Math.abs(v));
        if (max > 0) for (let i = 0; i < waveform.length; i++) waveform[i] *= (0.95 / max);

        return waveform;
    }
}

export class KittenTTS {
    constructor(config = {}) {
        this.config = {
            voiceUrl    : config.voiceUrl    ?? './default.bin',
            speed       : config.speed       ?? 1.0,
            debug       : config.debug       ?? false,
            concurrency : config.concurrency ?? 2,
            segmentMax  : config.segmentMax  ?? 220,
        };

        this._engine       = new _KittenEngine(this.config.debug);
        this._audioCtx     = null;
        this.isReady       = false;

        // --- Generation pipeline ---
        this._textQueue        = [];   // { text, genIndex }
        this._pendingAudio     = new Map(); // genIndex → AudioBuffer
        this._activeJobs       = 0;
        this._nextGenIndex     = 0;
        this._nextPlayIndex    = 0;

        // --- Playback ---
        this._isPlaying    = false;
        this._interrupted  = false;

        this._init();
    }

    log(msg, data = '') {
        if (this.config.debug) {
            console.log(`%c[KittenTTS] ${msg}`, 'color:#2563eb;font-weight:bold;', data);
        }
    }

    // -----------------------------------------------------------------------
    //  Init
    // -----------------------------------------------------------------------
    async _init() {
        this.log('Initializing…');

        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: _KITTENTTS_SAMPLE_RATE,
        });

        const ok = await this._engine.load(msg => this.log(msg));
        if (!ok) return;

        if (this.config.voiceUrl) {
            try {
                await this._engine.loadVoice(this.config.voiceUrl);
            } catch {
                this.log('⚠️ Voice load failed — continuing without voice');
            }
        }

        this.isReady = true;
        this.log('✅ Ready');
    }

    // -----------------------------------------------------------------------
    //  Public API
    // -----------------------------------------------------------------------

    /** Speak rawText.  Queues behind any audio already playing. */
    speak(rawText) {
        if (!this.isReady) { this.log('⚠️ Not ready yet'); return; }

        this._interrupted = false;

        const segments = this._parseText(rawText);
        this.log(`Queuing ${segments.length} segment(s)`);

        for (const text of segments) {
            this._textQueue.push({ text, genIndex: this._nextGenIndex++ });
        }

        this._pumpGenerators();
    }

    /**
     * Wipe the CacheStorage bucket so the model re-downloads on next load.
     * Returns a Promise<boolean> — true if the cache existed and was deleted.
     */
    clearModelCache() {
        return this._engine.clearModelCache();
    }

    /** Stop everything immediately and clear all queues. */
    interrupt() {
        this.log('🛑 Interrupt');
        this._interrupted = true;

        this._textQueue     = [];
        this._pendingAudio  = new Map();
        this._activeJobs    = 0;
        this._nextGenIndex  = 0;
        this._nextPlayIndex = 0;

        // Mute any audio that's mid-play
        this._audioCtx.suspend().then(() => {
            this._audioCtx.resume();
            this._isPlaying = false;
        });
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
            if (this._audioCtx.state === 'suspended') await this._audioCtx.resume();

            const waveform = await this._engine.generate(text, this.config.speed);

            if (this._interrupted) { this._activeJobs--; return; }

            const buffer = this._waveformToBuffer(waveform);
            this.log(`Segment [${genIndex}] ready (${buffer.duration.toFixed(2)}s)`);

            this._pendingAudio.set(genIndex, buffer);
            this._playNextInOrder();

        } catch (err) {
            this.log(`❌ Generation error [${genIndex}]: ${err.message}`);
        }

        this._activeJobs--;
        this._pumpGenerators(); // fill the slot we just freed
    }

    // -----------------------------------------------------------------------
    //  Internal: ordered playback
    // -----------------------------------------------------------------------

    _playNextInOrder() {
        if (this._isPlaying) return;
        if (!this._pendingAudio.has(this._nextPlayIndex)) return; // next not ready yet

        this._isPlaying = true;
        const buffer    = this._pendingAudio.get(this._nextPlayIndex);
        this._pendingAudio.delete(this._nextPlayIndex);
        this._nextPlayIndex++;

        const source = this._audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this._audioCtx.destination);

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

    /**
     * Splits arbitrary text into safe segments for KittenTTS:
     *  1. Strip markdown formatting
     *  2. Split on sentence-ending punctuation
     *  3. Further split long chunks at clause boundaries (, ; —)
     */
    _parseText(rawText) {
        // Strip common markdown
        let text = rawText
            .replace(/\*\*([^*]+)\*\*/g, '$1')
            .replace(/\*([^*]+)\*/g,     '$1')
            .replace(/`([^`]+)`/g,       '$1')
            .replace(/#{1,6}\s+/g,       '')
            .replace(/\s+/g,             ' ')
            .trim();

        // Protect common abbreviations
        const ABBREVS = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'St.', 'vs.', 'etc.'];
        const PLACEHOLDER = '\x00DOT\x00';
        for (const abbr of ABBREVS) {
            text = text.split(abbr).join(abbr.replace('.', PLACEHOLDER));
        }
        // Protect decimal numbers  e.g. "3.14"
        text = text.replace(/(\d+)\.(\d)/g, `$1${PLACEHOLDER}$2`);

        // Primary split: sentence-ending punctuation
        const raw = text.match(/[^.!?]+[.!?]+["'\])]?\s*|[^.!?]+$/g) ?? [text];

        const segments = [];
        for (const chunk of raw) {
            const s = chunk.replace(new RegExp(PLACEHOLDER, 'g'), '.').trim();
            if (!s) continue;

            if (s.length <= this.config.segmentMax) {
                segments.push(s);
            } else {
                // Secondary split at clause boundaries
                const parts = s
                    .split(/(?<=[,;—])\s+/)
                    .map(p => p.trim())
                    .filter(Boolean);

                // Merge very short parts back together to avoid tiny utterances
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

        return segments.filter(s => s.length > 0);
    }
}

window.initTTS = (config) => {
    window.tts       = new KittenTTS(config);
    window.speak     = (text) => window.tts.speak(text);
    window.interrupt = ()     => window.tts.interrupt();
};
