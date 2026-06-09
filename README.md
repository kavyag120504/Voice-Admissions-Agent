# BMU Voice Admissions Agent — Complete Documentation

> A real-time, voice-based AI admissions counselor for BML Munjal University (BMU).
> Built as an NLP project using Node.js, Python, Ollama (local LLM), Piper TTS, and a custom multi-layer NLP pipeline.
> Features a **trained ML intent classifier (TF-IDF + LogReg, 83% accuracy)** and a **child vs adult voice classifier (MFCC + SVM, 87.86% accuracy, macro F1 0.8786)**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Full NLP Pipeline](#3-full-nlp-pipeline)
4. [ML Intent Classification](#4-ml-intent-classification)
5. [Child vs Adult Voice Classifier](#5-child-vs-adult-voice-classifier)
6. [Python NLP Microservice](#6-python-nlp-microservice)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [Setup & Installation](#9-setup--installation)
10. [Running the Project](#10-running-the-project)
11. [Knowledge Base](#11-knowledge-base)
12. [Evaluation & Accuracy](#12-evaluation--accuracy)
13. [API Endpoints](#13-api-endpoints)
14. [Features](#14-features)

---

## 1. Project Overview

The BMU Advanced Call Agent is a **real-time voice conversational AI** that acts as a human-sounding admissions counselor named **"Aria"** for BML Munjal University. Students and parents can:

- **Speak** questions via microphone in English, Hindi, or Hinglish
- **Receive spoken answers** in a natural human voice (Piper TTS)
- **Interrupt** Aria mid-response (barge-in)
- Get instant answers about courses, fees, scholarships, placements, hostel, admission, safety, internships, and more

### What makes it an NLP project

The system implements a complete custom NLP pipeline:

- Spell correction (Levenshtein distance)
- Tokenization and stopword removal (English + Hinglish)
- Porter stemming
- OOV resolution (4-layer: direct map → stem → SoundEx phonetic → Levenshtein fuzzy)
- **ML-based intent classification** — custom trained TF-IDF + Logistic Regression classifier (83.08% accuracy, macro F1 0.84) with hybrid fallback to regex
- Named entity extraction (spaCy + custom rules)
- TF-IDF document retrieval
- Sentence-BERT semantic search + FAISS (Python microservice)
- Word2Vec embeddings for query expansion
- Context-aware dialogue management
- Hallucination detection
- Emotion-adaptive TTS

**Topic routing accuracy: 100% on 85 test cases**
**Intent classification accuracy: 83.08% (TF-IDF + Logistic Regression, trained on 1,001 samples)**

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Browser)                  │
│  🎤 Microphone → Web Speech API → Text                       │
│  🔊 Speaker ← Piper TTS Neural Voice ← WAV Audio            │
│  Phone-call UI with avatars, timer, transcript               │
└─────────────────────────────┬───────────────────────────────┘
                              │ WebSocket /realtime
┌─────────────────────────────▼───────────────────────────────┐
│              NODE.JS BACKEND (Port 3210)                     │
│                                                              │
│  NLP Pipeline → Knowledge Base → LLM → Piper TTS            │
│                                                              │
│  REST: /health /metrics /api/tts /api/asr/repair             │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
        ┌──────────▼──────┐    ┌──────────▼──────────┐
        │  bmu_facts.json  │    │  Python NLP Service  │
        │  25 topics EN+HI │    │  (Port 5001)         │
        │  Instant answers │    │  Sentence-BERT+FAISS │
        └──────────────────┘    │  Word2Vec, spaCy NER │
                                └─────────────────────┘
```

---

## 3. Full NLP Pipeline

Every user utterance goes through 14 stages:

| Stage | Module | What it does |
|---|---|---|
| 1 | Web Speech API | Converts voice to text |
| 2 | asrFallbackService.js | Fixes speech recognition errors |
| 3 | spellCorrectionService.js | Levenshtein spell correction |
| 4 | conversationMemoryService.js | Resolves follow-up context |
| 5 | entityExtractionService.js | Extracts program, %, year, flags |
| 6 | emotionService.js | Detects stressed/happy/angry/sad |
| 7 | nlpService.js | Tokenize, stopwords, concept map |
| 8 | nlpService.js | Porter stemming |
| 9 | nlpService.js | OOV resolution (4-layer) |
| 10 | nlpService.js | Intent detection (multi-label) |
| 11 | responsePolicy.js | Topic scoring + knowledge base |
| 12 | tfidfService.js + pythonNlpService.js | TF-IDF + Semantic search |
| 13 | llmService.js | phi3.5 LLM generation |
| 14 | hallucinationDetector.js + piperTtsService.js | Verify + speak |

---

## 4. ML Intent Classification

A custom **TF-IDF + Logistic Regression** classifier trained and evaluated entirely from scratch on real multilingual (English + Hinglish) admissions queries.

### Training Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **83.08%** |
| **Macro F1 Score** | **0.8354** |
| Weighted F1 Score | 0.8312 |
| CV Mean F1 (5-fold) | 0.8376 ± 0.0264 |
| Training samples | 800 (80% split) |
| Test samples | 201 (20% split) |
| Failure cases | 34 / 201 documented |

### Per-Class Breakdown

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| scholarship | 0.9545 | 0.9130 | **0.9333** |
| hostel | 0.8214 | 0.9583 | **0.8846** |
| fees | 0.8333 | 0.8696 | 0.8511 |
| admission | 0.9130 | 0.7778 | 0.8400 |
| location | **1.0000** | 0.7059 | 0.8276 |
| courses | 0.8333 | 0.7692 | 0.8000 |
| about | 0.6957 | 0.8889 | 0.7805 |
| placement | 0.8182 | 0.7200 | 0.7660 |

**Intent classes:** `fees` · `scholarship` · `placement` · `admission` · `courses` · `hostel` · `location` · `about`

**Dataset:** `backend/data/intent_dataset.csv` — 1,001 samples, 8 intent classes, multilingual (English + Hinglish)

**Model:** `nlp_service/train_intent_model.py` — TF-IDF (bigram, 8000 features) + Logistic Regression (C=5.0)

**Architecture:** Hybrid system — ML model runs first (confidence threshold ≥ 0.45), falls back to regex-based `detectIntent()` if unavailable or uncertain. Ensures zero degradation if Python service is offline.

### To retrain the model

```bash
cd nlp_service
python train_intent_model.py
```

Outputs: `intent_model.pkl`, `evaluation_report.txt`, `confusion_matrix.png`, `failure_cases.csv`

---

## 5. Child vs Adult Voice Classifier

A standalone **audio-based binary classifier** trained on real speech data from the **Mexican Emotional Speech Database (MESD)** to detect whether a speaker is a child or adult from their voice.

### Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **87.86%** |
| **Macro F1 Score** | **0.8786** |
| CV Mean F1 (5-fold) | 0.8652 ± 0.0175 |
| Training samples | 689 (80% split) |
| Test samples | 173 (20% split) |

### Per-Class Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| adult | 0.8824 | 0.8721 | 0.8772 |
| child | 0.8750 | 0.8851 | 0.8800 |

**Dataset:** MESD (Mexican Emotional Speech Database) — 862 WAV files, 430 adult / 432 child, perfectly balanced

**Labeling:** Decoded directly from filenames (`_A_` = adult, `_B_` = child) — no manual labeling needed

**Features extracted (per file):**
- 40 MFCC coefficients (mean + std) → 80 features
- 40 Delta-MFCC (mean + std) → 80 features — captures vocal tract dynamics
- 64 Mel-spectrogram bands (mean + std) → 128 features
- Fundamental frequency / pitch (mean, std, min, max) → 4 features — children have higher F0
- Zero-crossing rate, RMS energy, Spectral centroid → 6 features
- **Total: ~298 acoustic features per audio clip**

**Model:** SVM with RBF kernel (C=10, gamma=scale) + StandardScaler

**Key insight:** Children have higher fundamental frequency (F0), shorter vocal tract, and different spectral characteristics — all captured by MFCC + pitch features.

### To retrain

```bash
python audio_ml/train_child_adult_classifier.py
```

Outputs: `child_adult_model.pkl`, `child_adult_report.txt`, `child_adult_confusion_matrix.png`

---

## 6. Python NLP Microservice

Runs on **port 5001** alongside the Node.js backend.

| Feature | Technology | Purpose |
|---|---|---|
| Semantic Search | Sentence-BERT (all-MiniLM-L6-v2) + FAISS | Meaning-based retrieval |
| Named Entity Recognition | spaCy (en_core_web_sm) | Standard NER |
| Custom BMU NER | Rule-based | Program, %, year extraction |
| Word Embeddings | Word2Vec (Gensim) | Query expansion |

**Endpoints:**
- `GET /health` — service status (now includes `ml_intent_classifier` field)
- `POST /semantic-search` — Sentence-BERT + FAISS search
- `POST /ner` — spaCy + custom NER
- `POST /similar-words` — Word2Vec similarity
- `POST /analyze` — full NLP analysis
- `POST /predict-intent` — ML intent classification (TF-IDF + LogReg, returns intent + per-class confidence scores)

---

## 6. Tech Stack

| Component | Technology |
|---|---|
| Backend runtime | Node.js (ESM modules) |
| Web framework | Express.js |
| Real-time communication | WebSocket (ws library) |
| LLM | Ollama + phi3.5 (local, free) |
| TTS | Piper TTS (local neural voice — Amy en_US) |
| NLP library (Node) | natural (Porter stemmer, TF-IDF, SoundEx, Levenshtein) |
| Fuzzy matching | fastest-levenshtein |
| NLP microservice | Python Flask |
| Semantic search | Sentence-BERT + FAISS |
| Word embeddings | Word2Vec (Gensim) |
| Neural NER | spaCy |
| Speech input | Browser Web Speech API |
| Speech output | Web Audio API + Piper TTS |
| ML training | scikit-learn, HuggingFace transformers (Colab) |
| Data storage | JSON files |

---

## 7. Project Structure

```
bmu-advanced-call-agent/
├── backend/
│   ├── src/
│   │   ├── server.js                      # Express + WebSocket server
│   │   ├── config.js                      # Environment config
│   │   ├── realtime/
│   │   │   ├── pipeline.js                # Main NLP pipeline (15 stages)
│   │   │   └── sessionManager.js          # Session state
│   │   └── services/
│   │       ├── nlpService.js              # Tokenization, stemming, OOV, intent + detectIntentHybrid()
│   │       ├── spellCorrectionService.js  # Levenshtein spell correction
│   │       ├── entityExtractionService.js # Rule-based NER
│   │       ├── semanticSimilarityService.js # TF-IDF cosine similarity
│   │       ├── hallucinationDetector.js   # Fact verification
│   │       ├── responsePolicy.js          # Knowledge base topic matching
│   │       ├── tfidfService.js            # TF-IDF document retrieval
│   │       ├── llmService.js              # Ollama LLM integration
│   │       ├── piperTtsService.js         # Piper TTS (emotion-adaptive)
│   │       ├── pythonNlpService.js        # Python microservice client + predictIntentML()
│   │       ├── emotionService.js          # Emotion detection
│   │       ├── conversationMemoryService.js # Context-aware dialogue
│   │       ├── responseQualityService.js  # Quality enforcement
│   │       ├── runtimeMetricsService.js   # Live metrics
│   │       ├── sessionPersistenceService.js # Session persistence
│   │       ├── knowledgeRetrievalService.js # Legacy retrieval
│   │       └── asrFallbackService.js      # ASR repair
│   ├── public/
│   │   ├── index.html                     # Phone-call UI
│   │   └── app.js                         # Frontend logic
│   ├── data/
│   │   ├── bmu_facts.json                 # Knowledge base (25 topics, EN+HI)
│   │   ├── intent_dataset.csv             # ML training data (1,001 samples)
│   │   └── session_store.json             # Persisted sessions
│   ├── .env                               # Environment variables (not in git)
│   └── package.json
├── nlp_service/
│   ├── nlp_server.py                      # Python Flask NLP microservice
│   ├── train_intent_model.py              # ML training + evaluation script
│   ├── intent_model.pkl                   # Trained model (generated, not in git)
│   └── requirements.txt                   # Python dependencies
└── README.md
```

---

## 8. Setup & Installation

### Prerequisites

- Node.js v18+
- Python 3.10+
- Ollama (https://ollama.ai/download)
- Piper TTS (https://github.com/rhasspy/piper/releases/tag/2023.11.14-2)
- Chrome browser

### Step 1 — Install Node.js dependencies

```bash
cd backend
npm install
```

### Step 2 — Install Python dependencies

```bash
cd nlp_service
pip install flask flask-cors sentence-transformers faiss-cpu spacy gensim numpy scikit-learn pandas matplotlib seaborn
python -m spacy download en_core_web_sm
```

### Step 3 — Train the ML Intent Classifier

```bash
cd nlp_service
python train_intent_model.py
```

This generates `intent_model.pkl`, `evaluation_report.txt`, `confusion_matrix.png`, and `failure_cases.csv`.

### Step 3 — Install Ollama and pull model

```bash
ollama pull phi3.5
```

### Step 4 — Setup Piper TTS

1. Download `piper_windows_amd64.zip` and extract to `D:\piper_windows_amd64\piper\`
2. Download voice model:
   ```
   https://huggingface.co/diffusionstudio/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
   ```
3. Download config:
   ```
   https://huggingface.co/diffusionstudio/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
   ```

### Step 5 — Configure environment

Create `.env` in `backend/`:

```env
PORT=3210
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5:latest
PREFERRED_OLLAMA_MODELS=phi3.5:latest,phi3.5,llama3.2:3b
MAX_CONTEXT_TURNS=14
REQUEST_TIMEOUT_MS=30000
LLM_MAX_SILENCE_MS=25000
STRICT_DEMO_MODE=false
MAX_RESPONSE_SENTENCES=3
```

---

## 9. Running the Project

You need **two terminals**:

**Terminal 1 — Python NLP microservice:**
```bash
cd bmu-advanced-call-agent
python nlp_service/nlp_server.py
```

**Terminal 2 — Node.js backend:**
```bash
cd bmu-advanced-call-agent/backend
node src/server.js
```

**Open browser:** http://localhost:3210

**Verify:**
- Health: http://localhost:3210/health
- Piper TTS: http://localhost:3210/api/tts/status
- Python NLP: http://localhost:5001/health
- Metrics: http://localhost:3210/metrics

---

## 10. Knowledge Base

**25 topics** in `bmu_facts.json`, each with English + Hindi conversational answers:

| Topic | Description |
|---|---|
| accreditation | UGC recognition, NAAC grade, approvals |
| why_bmu | What makes BMU unique, worth the fees |
| safety_campus | Campus security, CCTV, attendance |
| extracurricular | Clubs, fests, exchange programs |
| internship_industry | Internships, Hero Group, industry tie-ups |
| university_overview | About BMU, rankings, Hero Group |
| courses_overview | All programs offered |
| btech_overview | BTech branches and specializations |
| btech_cse | CSE details, fees, placements |
| btech_mechanical_ecome | Mechanical and EComE |
| btech_fees | Exact BTech fee structure |
| mba_program | MBA details and fees |
| law_programs | BA LLB, BBA LLB, LLB |
| bba_bcom | BBA, B.Com, Liberal Arts |
| scholarships | All scholarship criteria |
| hostel | Hostel facilities and charges |
| placements | Placement stats and recruiters |
| admission_process | Step-by-step admission guide |
| eligibility_btech | BTech eligibility |
| documents_required | Required documents |
| campus_facilities | Campus infrastructure |
| contact_info | Location, address, contact |
| education_loan | Loan assistance |
| phd_mtech | PhD and M.Tech |
| total_fees_with_hostel | All-inclusive cost |

---

## 11. Evaluation & Accuracy

### ML Intent Classification (Local — TF-IDF + Logistic Regression)

| Metric | Value |
|---|---|
| **Test Accuracy** | **83.08%** |
| **Macro F1** | **0.8354** |
| CV Mean F1 (5-fold) | 0.8376 ± 0.0264 |
| Training samples | 800 |
| Test samples | 201 |

Run `python nlp_service/train_intent_model.py` to reproduce all results.

### Topic Routing (Production Pipeline)

**100% on 85 test cases** across 9 categories

| Category | Accuracy |
|---|---|
| Fees | 100% |
| Scholarships | 100% |
| Placements | 100% |
| Courses | 100% |
| Admission | 100% |
| Hostel | 100% |
| Location | 100% |
| About BMU | 100% |
| Spell Correction | 100% |

### Response Latency

| Type | Latency |
|---|---|
| Greeting / Small talk | ~20ms |
| Grounded answer | ~35ms |
| LLM generated | 5–30s |

---

## 12. API Endpoints

### Node.js (Port 3210)

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Server status + model |
| /metrics | GET | Live usage stats |
| /api/tts | POST | Piper TTS synthesis |
| /api/tts/status | GET | Piper availability |
| /api/asr/repair | POST | Transcript repair |
| /realtime | WebSocket | Real-time conversation |

### Python NLP (Port 5001)

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Service status (includes ml_intent_classifier) |
| /semantic-search | POST | Sentence-BERT + FAISS |
| /ner | POST | spaCy + custom NER |
| /similar-words | POST | Word2Vec similarity |
| /analyze | POST | Full NLP analysis |
| /predict-intent | POST | ML intent classification (TF-IDF + LogReg) |

---

## 13. Features

### Voice
- Real-time mic input (Web Speech API)
- Language auto-detection and locking (EN/HI/Hinglish)
- Barge-in interruption
- Echo detection
- Done Speaking button + 3-second silence auto-send
- Sentence-level streaming TTS
- Emotion-adaptive speech rate (Piper)

### Conversation
- Human-sounding responses (Aria persona)
- Handles greetings, small talk, thank you, bye, who are you, are you AI
- Context-aware follow-up queries
- Bilingual responses (English + Hindi/Hinglish)
- Session persistence across restarts

### NLP
- 4-layer OOV resolution
- 300+ Hinglish concept mappings
- **Custom trained ML intent classifier** (TF-IDF + Logistic Regression, 83.08% accuracy, macro F1 0.84)
- **Hybrid intent detection** — ML primary, regex fallback (zero degradation if offline)
- **Evaluation pipeline** — Precision, Recall, F1, Confusion Matrix, failure case analysis
- spaCy NER + custom entity extraction
- TF-IDF + Sentence-BERT semantic search
- Word2Vec query expansion
- Hallucination detection
- Spell correction

### UI
- Phone call interface (two avatars, call timer)
- Speaking/thinking/listening animations
- Transcript with message bubbles
- Typing indicator
- Done Speaking + Interrupt buttons
