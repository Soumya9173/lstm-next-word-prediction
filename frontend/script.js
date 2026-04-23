/**
 * NextWord AI — Frontend Logic
 * Real-time LSTM next-word prediction with debounced API calls.
 */

(() => {
    "use strict";

    // ── Configuration ────────────────────────────────────────
    const API_BASE = window.location.origin;
    const DEBOUNCE_MS = 350;

    // ── DOM References ───────────────────────────────────────
    const $input       = document.getElementById("input-text");
    const $ghostText   = document.getElementById("ghost-text");
    const $genBtn      = document.getElementById("generate-btn");
    const $copyBtn     = document.getElementById("copy-btn");
    const $clearBtn    = document.getElementById("clear-btn");
    const $wordCount   = document.getElementById("word-count");
    const $wordValue   = document.getElementById("word-count-value");
    const $predList    = document.getElementById("predictions-list");
    const $predEmpty   = document.getElementById("predictions-empty");
    const $predLoading = document.getElementById("predictions-loading");
    const $predSub     = document.getElementById("predictions-subtitle");
    const $statVocab   = document.getElementById("stat-vocab");
    const $statMaxlen  = document.getElementById("stat-maxlen");
    const $statPreds   = document.getElementById("stat-predictions");
    const $statLatency = document.getElementById("stat-latency");
    const $statusBadge = document.getElementById("status-badge");
    const $histList    = document.getElementById("history-list");
    const $histEmpty   = document.getElementById("history-empty");
    const $clearHist   = document.getElementById("clear-history-btn");
    const $toastWrap   = document.getElementById("toast-container");
    const $temperature = document.getElementById("temperature");
    const $tempValue   = document.getElementById("temp-value");
    const $samplingMode = document.getElementById("sampling-mode");

    // ── State ────────────────────────────────────────────────
    let predictionCount = 0;
    let currentSuggestion = "";
    let debounceTimer = null;
    let isGenerating = false;

    // ── Utilities ────────────────────────────────────────────

    /** Show a toast notification */
    function toast(message, type = "info") {
        const el = document.createElement("div");
        el.className = `toast toast--${type}`;
        el.textContent = message;
        $toastWrap.appendChild(el);
        setTimeout(() => {
            el.style.animation = "toast-out 0.3s ease forwards";
            setTimeout(() => el.remove(), 300);
        }, 3000);
    }

    /** Format large numbers: 8978 → 8.9K */
    function formatNum(n) {
        if (n >= 1000) return (n / 1000).toFixed(1) + "K";
        return String(n);
    }

    /** Get current sampling parameters from UI controls */
    function getSamplingParams() {
        const temperature = parseFloat($temperature.value);
        const mode = $samplingMode.value;
        let top_k = 0;
        let top_p = 1.0;
        if (mode === "top-k") top_k = 40;
        if (mode === "top-p") top_p = 0.9;
        if (mode === "top-k-p") { top_k = 40; top_p = 0.9; }
        return { temperature, top_k, top_p };
    }

    // ── API Calls ────────────────────────────────────────────

    async function apiHealth() {
        const res = await fetch(`${API_BASE}/api/health`);
        if (!res.ok) throw new Error("Health check failed");
        return res.json();
    }

    async function apiPredictTop(text, topK = 5) {
        const res = await fetch(`${API_BASE}/api/predict/top`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, top_k: topK }),
        });
        if (!res.ok) throw new Error("Prediction failed");
        return res.json();
    }

    async function apiGenerate(text, nWords) {
        const sampling = getSamplingParams();
        const res = await fetch(`${API_BASE}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, n_words: nWords, ...sampling }),
        });
        if (!res.ok) throw new Error("Generation failed");
        return res.json();
    }

    // ── Health Check ─────────────────────────────────────────

    async function checkHealth() {
        try {
            const data = await apiHealth();
            $statusBadge.innerHTML = `<span class="badge__dot"></span>Model Active`;
            $statusBadge.className = "badge badge--live";
            $statVocab.textContent = formatNum(data.vocab_size);
            $statMaxlen.textContent = data.max_len;
            $genBtn.disabled = false;
            toast("Model loaded successfully!", "success");
        } catch {
            $statusBadge.innerHTML = `<span class="badge__dot"></span>Offline`;
            $statusBadge.className = "badge badge--error";
            toast("Cannot connect to the server. Is it running?", "error");
        }
    }

    // ── Prediction Logic ─────────────────────────────────────

    function showLoading() {
        $predEmpty.style.display = "none";
        $predList.style.display = "none";
        $predLoading.style.display = "flex";
    }

    function showPredictions(predictions) {
        $predLoading.style.display = "none";
        $predEmpty.style.display = "none";
        $predList.style.display = "flex";
        $predList.innerHTML = "";

        if (!predictions || predictions.length === 0) {
            showEmpty();
            return;
        }

        predictions.forEach((p, i) => {
            const chip = document.createElement("button");
            chip.className = "prediction-chip";
            chip.innerHTML = `
                <span>${escapeHtml(p.word)}</span>
                <span class="prediction-chip__confidence">${p.confidence}%</span>
            `;
            chip.title = `Click to add "${p.word}"`;
            chip.addEventListener("click", () => acceptWord(p.word));
            $predList.appendChild(chip);
        });

        // Update ghost text with top suggestion
        currentSuggestion = predictions[0]?.word || "";
        updateGhostText();

        $predSub.textContent = `${predictions.length} suggestions · Click to accept`;
    }

    function showEmpty() {
        $predLoading.style.display = "none";
        $predList.style.display = "none";
        $predEmpty.style.display = "flex";
        $predSub.textContent = "Top suggestions will appear here";
        currentSuggestion = "";
        updateGhostText();
    }

    function updateGhostText() {
        if (!currentSuggestion || !$input.value.trim()) {
            $ghostText.innerHTML = "";
            return;
        }
        // Show the typed text (invisible) + suggestion in ghost color
        const typed = $input.value;
        $ghostText.innerHTML =
            `<span style="visibility:hidden">${escapeHtml(typed)}</span><span class="ghost-suggestion"> ${escapeHtml(currentSuggestion)}</span>`;
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    async function fetchPredictions() {
        const text = $input.value.trim();
        if (!text) {
            showEmpty();
            return;
        }

        showLoading();

        const start = performance.now();
        try {
            const data = await apiPredictTop(text, 5);
            const latency = Math.round(performance.now() - start);
            $statLatency.textContent = latency + "ms";

            predictionCount++;
            $statPreds.textContent = predictionCount;

            showPredictions(data.predictions);
        } catch (err) {
            showEmpty();
            console.error("Prediction error:", err);
        }
    }

    function debouncedPredict() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(fetchPredictions, DEBOUNCE_MS);
    }

    // ── Accept Word ──────────────────────────────────────────

    function acceptWord(word) {
        const currentText = $input.value;
        const needsSpace = currentText.length > 0 && !currentText.endsWith(" ");
        $input.value = currentText + (needsSpace ? " " : "") + word + " ";
        $input.focus();
        currentSuggestion = "";
        updateGhostText();
        // Trigger new prediction
        debouncedPredict();
    }

    // ── Generate ─────────────────────────────────────────────

    async function handleGenerate() {
        const text = $input.value.trim();
        if (!text || isGenerating) return;

        const nWords = parseInt($wordCount.value, 10);
        isGenerating = true;
        $genBtn.classList.add("btn--loading");
        $genBtn.disabled = true;

        const start = performance.now();
        try {
            const data = await apiGenerate(text, nWords);
            const latency = Math.round(performance.now() - start);
            $statLatency.textContent = latency + "ms";

            $input.value = data.generated_text;
            predictionCount++;
            $statPreds.textContent = predictionCount;

            // Add to history
            addHistory(text, data.generated_text, data.words_added);

            toast(`Generated ${data.words_added} words`, "success");

            // Trigger new prediction for the extended text
            debouncedPredict();
        } catch (err) {
            toast("Generation failed. Check the server.", "error");
            console.error("Generate error:", err);
        } finally {
            isGenerating = false;
            $genBtn.classList.remove("btn--loading");
            $genBtn.disabled = false;
        }
    }

    // ── History ──────────────────────────────────────────────

    function addHistory(input, generated, wordsAdded) {
        $histEmpty.style.display = "none";
        const addedText = generated.slice(input.length).trim();
        const item = document.createElement("div");
        item.className = "history-item";
        item.innerHTML = `
            <span class="history-item__input">${escapeHtml(input)}</span>
            <span class="history-item__generated"> ${escapeHtml(addedText)}</span>
            <div class="history-item__meta">+${wordsAdded} words · ${new Date().toLocaleTimeString()}</div>
        `;
        // Click to load into textarea
        item.style.cursor = "pointer";
        item.title = "Click to load this text";
        item.addEventListener("click", () => {
            $input.value = generated;
            $input.focus();
            debouncedPredict();
        });
        $histList.prepend(item);
    }

    function clearHistory() {
        $histList.innerHTML = "";
        $histEmpty.style.display = "block";
    }

    // ── Event Listeners ──────────────────────────────────────

    // Real-time prediction on typing
    $input.addEventListener("input", () => {
        $genBtn.disabled = !$input.value.trim();
        debouncedPredict();
    });

    // Tab to accept ghost suggestion
    $input.addEventListener("keydown", (e) => {
        if (e.key === "Tab" && currentSuggestion) {
            e.preventDefault();
            acceptWord(currentSuggestion);
        }
        // Ctrl+Enter to generate
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleGenerate();
        }
    });

    // Update ghost positioning when textarea scrolls
    $input.addEventListener("scroll", () => {
        $ghostText.scrollTop = $input.scrollTop;
    });

    // Buttons
    $genBtn.addEventListener("click", handleGenerate);

    $copyBtn.addEventListener("click", () => {
        const text = $input.value.trim();
        if (!text) {
            toast("Nothing to copy", "error");
            return;
        }
        navigator.clipboard.writeText(text).then(() => {
            toast("Copied to clipboard!", "success");
        }).catch(() => {
            toast("Failed to copy", "error");
        });
    });

    $clearBtn.addEventListener("click", () => {
        $input.value = "";
        $genBtn.disabled = true;
        showEmpty();
        $input.focus();
    });

    $wordCount.addEventListener("input", () => {
        $wordValue.textContent = $wordCount.value;
    });

    $clearHist.addEventListener("click", clearHistory);

    $temperature.addEventListener("input", () => {
        $tempValue.textContent = parseFloat($temperature.value).toFixed(1);
    });

    $samplingMode.addEventListener("change", () => {
        const mode = $samplingMode.value;
        const labels = { "greedy": "Greedy", "top-k": "Top-K", "top-p": "Top-P", "top-k-p": "Top-K + Top-P" };
        toast(`Sampling: ${labels[mode]}`, "info");
    });

    // ── Init ─────────────────────────────────────────────────

    checkHealth();
})();
