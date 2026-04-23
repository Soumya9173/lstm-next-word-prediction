# 🧠 NextWord AI — LSTM Next Word Prediction

A full-stack web application that predicts the next word in real-time using an LSTM (Long Short-Term Memory) neural network trained on a quotes dataset.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## ✨ Features

- **Real-time Prediction** — Predictions appear as you type with 350ms debounce
- **Top-K Suggestions** — See the top 5 next-word candidates with confidence scores
- **Ghost Text** — Subtle inline suggestion overlay (press Tab to accept)
- **Multi-Word Generation** — Generate up to 20 words at once
- **Generation History** — Track and reload previous generations
- **Live Stats** — Vocabulary size, max sequence length, prediction count, latency
- **Premium Dark UI** — Glassmorphism, micro-animations, responsive design

---

## 🏗️ Architecture

```
┌─────────────┐      HTTP/JSON       ┌──────────────┐
│   Frontend   │  ◄──────────────►   │   FastAPI     │
│  HTML/CSS/JS │                     │   Backend     │
└─────────────┘                      └──────┬───────┘
                                            │
                                     ┌──────▼───────┐
                                     │  LSTM Model   │
                                     │  TensorFlow   │
                                     └──────────────┘
```

---

## 📂 Project Structure

```
RNN project/
├── backend/
│   ├── app.py                  # FastAPI server
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── index.html              # Main page
│   ├── style.css               # Premium dark theme
│   └── script.js               # Frontend logic
├── models/
│   ├── lstm_model.keras         # Trained LSTM model
│   ├── tokinizer.pkl            # Keras tokenizer
│   └── max_len.pkl              # Max sequence length
├── data/
│   └── qoute_dataset.csv       # Quotes dataset (3,038 quotes)
├── notebooks/
│   └── Untitled3.ipynb          # Training notebook
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nextword-ai.git
cd nextword-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run the server

```bash
cd backend
python app.py
```

### 5. Open in browser

Navigate to **http://127.0.0.1:8000** and start typing!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & model info |
| `POST` | `/api/predict` | Predict single next word |
| `POST` | `/api/predict/top` | Top-K predictions with confidence |
| `POST` | `/api/generate` | Generate multiple words |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/predict/top \
  -H "Content-Type: application/json" \
  -d '{"text": "the world is", "top_k": 5}'
```

### Example Response

```json
{
  "predictions": [
    {"word": "a", "confidence": 12.34},
    {"word": "not", "confidence": 8.21},
    {"word": "full", "confidence": 5.67}
  ],
  "input_text": "the world is"
}
```

---

## 🧬 Model Details

| Property | Value |
|----------|-------|
| Architecture | Embedding → LSTM → Dense |
| Vocabulary Size | 10,000 |
| Embedding Dimension | 50 |
| LSTM Units | 128 |
| Training Dataset | 3,038 famous quotes |
| Training Samples | 85,271 subsequences |
| Epochs Trained | 100 |
| Framework | TensorFlow / Keras |

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, TensorFlow, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Model**: LSTM (Keras Sequential)
- **Fonts**: Inter, JetBrains Mono

---

## 📈 Future Improvements

- [ ] Retrain with Dropout regularization to reduce overfitting
- [ ] Add Early Stopping and learning rate scheduling
- [ ] Use pre-trained word embeddings (GloVe / Word2Vec)
- [ ] Add temperature-based sampling for more creative outputs
- [ ] Deploy to cloud (Render / Railway / AWS)
- [ ] Add user authentication and saved sessions

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
