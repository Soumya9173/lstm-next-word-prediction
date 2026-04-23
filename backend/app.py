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

class PredictResponse(BaseModel):
    next_word: str
    input_text: str

class GenerateRequest(BaseModel):
    text: str
    n_words: int = 5

class GenerateResponse(BaseModel):
    generated_text: str
    input_text: str
    words_added: int

class TopPredictionsRequest(BaseModel):
    text: str
    top_k: int = 5

class TopPredictionsResponse(BaseModel):
    predictions: list
    input_text: str

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_next_word(text: str) -> str:
    """Predict the single most likely next word."""
    text = text.lower().strip()
    if not text:
        return ""
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return ""
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    pred = model.predict(seq, verbose=0)
    idx = int(np.argmax(pred))
    return index_to_word.get(idx, "")


def _predict_top_k(text: str, k: int = 5) -> list:
    """Return top-k predicted words with confidence scores."""
    text = text.lower().strip()
    if not text:
        return []
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return []
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    pred = model.predict(seq, verbose=0)[0]
    top_indices = pred.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        word = index_to_word.get(int(idx), "")
        if word:
            results.append({
                "word": word,
                "confidence": round(float(pred[idx]) * 100, 2),
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
    """Predict the single most likely next word."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    word = _predict_next_word(req.text)
    return PredictResponse(next_word=word, input_text=req.text)


@app.post("/api/predict/top", response_model=TopPredictionsResponse)
def predict_top(req: TopPredictionsRequest):
    """Return top-k next word predictions with confidence."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    preds = _predict_top_k(req.text, req.top_k)
    return TopPredictionsResponse(predictions=preds, input_text=req.text)


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate multiple next words sequentially."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    n = min(req.n_words, 50)  # cap at 50 words
    current = req.text
    words_added = 0
    for _ in range(n):
        word = _predict_next_word(current)
        if not word:
            break
        current += " " + word
        words_added += 1
    return GenerateResponse(
        generated_text=current,
        input_text=req.text,
        words_added=words_added,
    )


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
