# 🧠 NextWord AI — LSTM Next Word Prediction

A full-stack web application that predicts the next word in real-time using an LSTM (Long Short-Term Memory) neural network trained on a quotes dataset.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## ✨ Features

- **Real-time Prediction** — Predictions appear as you type with 350ms debounce
- **Advanced Generative Sampling** — Control creativity using Temperature, Top-K, and Top-P (Nucleus) sampling
- **Mid-Word Autocomplete** — Suggests vocabulary word completions while you are still typing a word
- **Top-K Suggestions** — See the top 5 next-word candidates with confidence scores
- **Ghost Text** — Subtle inline suggestion overlay (press Tab to accept)
- **Multi-Word Generation** — Generate up to 50 words at once
- **Generation History** — Track and reload previous generations
- **Live Stats** — Vocabulary size, max sequence length, prediction count, latency
- **Light & Dark Themes** — Custom glassmorphic UI with local storage persistence and a seamless toggle

---

## 📸 UI Screenshot

> **Note to developer:** *Drop a screenshot of your app here! If you are editing on GitHub, you can just drag and drop an image file right here.*
![NextWord AI UI Placeholder](https://via.placeholder.com/800x450.png?text=NextWord+AI+Web+Interface)

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

## 🎲 How Sampling Works

The application implements industry-standard sampling techniques to make the AI's text generation feel more natural and creative. Without sampling, the model would always pick the exact same "most likely" word (Greedy Search), resulting in boring, repetitive loops.

- **Temperature**: Controls the randomness of predictions. 
  - `Low (e.g., 0.2)`: Very conservative, picks highly probable words. Good for factual or predictable text.
  - `1.0 (Default)`: Uses the model's raw probabilities.
  - `High (e.g., 1.5)`: Flattens the probabilities, giving rare words a higher chance. Good for creative/wild text.
- **Top-K Filtering**: Limits the AI to only choose from the top *K* (e.g., 40) most likely next words, discarding the long tail of highly improbable words to prevent absolute gibberish.
- **Top-P (Nucleus) Filtering**: A smarter alternative to Top-K. It dynamically shrinks or expands the pool of candidate words based on their cumulative probability. If the model is highly confident, it might only pick from 2 words. If uncertain, it might pick from 50. It caps the cumulative probability at *P* (e.g., 0.9 or 90%).

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
- [ ] Deploy to cloud (Render / Railway / AWS)
- [ ] Add user authentication and saved sessions

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
