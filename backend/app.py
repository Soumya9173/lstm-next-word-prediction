"""
LSTM Next Word Prediction — FastAPI Backend
Serves the trained LSTM model for real-time next word prediction.
"""

import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ---------------------------------------------------------------------------
# Load model & artefacts once at startup
# ---------------------------------------------------------------------------
print("Loading LSTM model...")
model = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
print("Model loaded [OK]")

with open(os.path.join(MODEL_DIR, "tokinizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded [OK]")

with open(os.path.join(MODEL_DIR, "max_len.pkl"), "rb") as f:
    max_len = pickle.load(f)
print(f"Max sequence length: {max_len}")

# Build reverse word index
word_index = tokenizer.word_index
index_to_word = {idx: word for word, idx in word_index.items()}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Next Word Predictor",
    description="LSTM-powered next-word prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str
    temperature: float = 1.0  # 0.1 = conservative, 2.0 = creative
    top_k: int = 0            # 0 = disabled, else keep top K words
    top_p: float = 1.0        # 1.0 = disabled, 0.9 = nucleus sampling

class PredictResponse(BaseModel):
    next_word: str
    input_text: str
    sampling: str

class GenerateRequest(BaseModel):
    text: str
    n_words: int = 5
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

class GenerateResponse(BaseModel):
    generated_text: str
    input_text: str
    words_added: int
    sampling: str

class TopPredictionsRequest(BaseModel):
    text: str
    top_k: int = 5

class TopPredictionsResponse(BaseModel):
    predictions: list
    input_text: str

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _get_probs(text: str):
    """Get raw probability distribution for next word."""
    text = text.lower().strip()
    if not text:
        return None
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return None
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    return model.predict(seq, verbose=0)[0]


def _sample_from_probs(probs, temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample a word index from the probability distribution using
    temperature scaling, top-k filtering, and top-p (nucleus) filtering.
    """
    logits = np.log(probs + 1e-10)  # convert to log-probs

    # --- Temperature scaling ---
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    # Re-exponentiate to get scaled probabilities
    exp_logits = np.exp(logits - np.max(logits))  # subtract max for stability
    scaled_probs = exp_logits / exp_logits.sum()

    # --- Top-K filtering ---
    if top_k > 0:
        top_k = min(top_k, len(scaled_probs))
        top_k_indices = np.argsort(scaled_probs)[-top_k:]
        mask = np.zeros_like(scaled_probs)
        mask[top_k_indices] = 1
        scaled_probs = scaled_probs * mask
        scaled_probs = scaled_probs / scaled_probs.sum()

    # --- Top-P (nucleus) filtering ---
    if top_p < 1.0:
        sorted_indices = np.argsort(scaled_probs)[::-1]
        sorted_probs = scaled_probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        # Find cutoff index where cumulative probability exceeds top_p
        cutoff = np.searchsorted(cumulative, top_p) + 1
        allowed_indices = sorted_indices[:cutoff]
        mask = np.zeros_like(scaled_probs)
        mask[allowed_indices] = 1
        scaled_probs = scaled_probs * mask
        scaled_probs = scaled_probs / scaled_probs.sum()

    # --- Sample ---
    idx = np.random.choice(len(scaled_probs), p=scaled_probs)
    return int(idx)


def _predict_next_word(text: str, temperature=1.0, top_k=0, top_p=1.0) -> str:
    """Predict next word using sampling strategy."""
    probs = _get_probs(text)
    if probs is None:
        return ""
    # Greedy mode: no sampling, just argmax
    if temperature == 1.0 and top_k == 0 and top_p == 1.0:
        idx = int(np.argmax(probs))
    else:
        idx = _sample_from_probs(probs, temperature, top_k, top_p)
    return index_to_word.get(idx, "")


def _get_sampling_label(temperature, top_k, top_p) -> str:
    """Return a human-readable label for the sampling strategy."""
    if temperature == 1.0 and top_k == 0 and top_p == 1.0:
        return "greedy"
    parts = []
    if temperature != 1.0:
        parts.append(f"temp={temperature}")
    if top_k > 0:
        parts.append(f"top-k={top_k}")
    if top_p < 1.0:
        parts.append(f"top-p={top_p}")
    return ", ".join(parts)


def _predict_top_k_list(text: str, k: int = 5) -> list:
    """Return top-k predicted words with confidence scores."""
    probs = _get_probs(text)
    if probs is None:
        return []
    top_indices = probs.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        word = index_to_word.get(int(idx), "")
        if word:
            results.append({
                "word": word,
                "confidence": round(float(probs[idx]) * 100, 2),
            })
    return results

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "model": "lstm_model.keras",
        "vocab_size": len(word_index),
        "max_len": int(max_len),
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict the single most likely next word with optional sampling."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    word = _predict_next_word(req.text, req.temperature, req.top_k, req.top_p)
    label = _get_sampling_label(req.temperature, req.top_k, req.top_p)
    return PredictResponse(next_word=word, input_text=req.text, sampling=label)


@app.post("/api/predict/top", response_model=TopPredictionsResponse)
def predict_top(req: TopPredictionsRequest):
    """Return top-k next word predictions with confidence."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    preds = _predict_top_k_list(req.text, req.top_k)
    return TopPredictionsResponse(predictions=preds, input_text=req.text)


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate multiple next words with optional sampling."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    n = min(req.n_words, 50)  # cap at 50 words
    current = req.text
    words_added = 0
    for _ in range(n):
        word = _predict_next_word(current, req.temperature, req.top_k, req.top_p)
        if not word:
            break
        current += " " + word
        words_added += 1
    label = _get_sampling_label(req.temperature, req.top_k, req.top_p)
    return GenerateResponse(
        generated_text=current,
        input_text=req.text,
        words_added=words_added,
        sampling=label,
    )


@app.get("/api/autocomplete")
def autocomplete(prefix: str = "", limit: int = 8):
    """Return vocabulary words matching a prefix for mid-word completion."""
    prefix = prefix.lower().strip()
    if not prefix or len(prefix) < 1:
        return {"suggestions": [], "prefix": prefix}
    matches = []
    for word in word_index:
        if word.startswith(prefix) and word != prefix:
            matches.append(word)
            if len(matches) >= limit:
                break
    return {"suggestions": matches, "prefix": prefix}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
