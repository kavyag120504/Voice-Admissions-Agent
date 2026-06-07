# BMU Advanced Call Agent — Complete Documentation

> A real-time, voice-based AI admissions counselor for BML Munjal University (BMU).
> Built as an NLP project using Node.js, Python, Ollama (local LLM), Piper TTS, and a custom multi-layer NLP pipeline with ML-based intent classification.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Full NLP Pipeline](#3-full-nlp-pipeline)
4. [ML Intent Classification](#4-ml-intent-classification)
5. [Python NLP Microservice](#5-python-nlp-microservice)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Project](#9-running-the-project)
10. [Knowledge Base](#10-knowledge-base)
11. [Evaluation & Accuracy](#11-evaluation--accuracy)
12. [API Endpoints](#12-api-endpoints)
13. [Features](#13-features)

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
- **ML-based intent classification** (Logistic Regression, SVM, DistilBERT — 93.60% accuracy)
- Named entity extraction (spaCy + custom rules)
- TF-IDF document retrieval
- Sentence-BERT semantic search + FAISS (Python microservice)
- Word2Vec embeddings for query expansion
- Context-aware dialogue managementiThree models were trained and tested on 856 labeled queries containing 8  classes of intent with an 80 train and 20 test split.

- Hallucination detection
- Emotion-adaptive TTS

**Topic routing accuracy: 100% on 85 test cases**
**Intent classification accuracy: 93.60% (DistilBERT)**

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

Three models trained and evaluated on **1001 labeled queries** across **8 intent classes**:

| Model | Accuracy |
|---|---|
| Logistic Regression (TF-IDF) | 89.53% |
| SVM (TF-IDF) | 90.70% |
| **DistilBERT (Fine-tuned)** | **93.60%** |

**Intent classes:** `fees` · `scholarship` · `placement` · `admission` · `courses` · `hostel` · `location` · `about`

**Dataset:** `data/intent_dataset.csv` — 1001 samples, balanced across 8 classes

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC (all classes > 0.97)

---

## 5. Python NLP Microservice

Runs on **port 5001** alongside the Node.js backend.

| Feature | Technology | Purpose |
|---|---|---|
| Semantic Search | Sentence-BERT (all-MiniLM-L6-v2) + FAISS | Meaning-based retrieval |
| Named Entity Recognition | spaCy (en_core_web_sm) | Standard NER |
| Custom BMU NER | Rule-based | Program, %, year extraction |
| Word Embeddings | Word2Vec (Gensim) | Query expansion |

**Endpoints:**
- `GET /health` — service status
- `POST /semantic-search` — Sentence-BERT + FAISS search
- `POST /ner` — spaCy + custom NER
- `POST /similar-words` — Word2Vec similarity
- `POST /analyze` — full NLP analysis

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
│   │   │   ├── pipeline.js                # Main NLP pipeline (14 stages)
│   │   │   └── sessionManager.js          # Session state
│   │   └── services/
│   │       ├── nlpService.js              # Tokenization, stemming, OOV, intent
│   │       ├── spellCorrectionService.js  # Levenshtein spell correction
│   │       ├── entityExtractionService.js # Rule-based NER
│   │       ├── semanticSimilarityService.js # TF-IDF cosine similarity
│   │       ├── hallucinationDetector.js   # Fact verification
│   │       ├── responsePolicy.js          # Knowledge base topic matching
│   │       ├── tfidfService.js            # TF-IDF document retrieval
│   │       ├── llmService.js              # Ollama LLM integration
│   │       ├── piperTtsService.js         # Piper TTS (emotion-adaptive)
│   │       ├── pythonNlpService.js        # Python microservice client
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
│   │   ├── intent_dataset.csv             # ML training data (1001 samples)
│   │   └── session_store.json             # Persisted sessions
│   ├── .env                               # Environment variables
│   └── package.json
├── nlp_service/
│   ├── nlp_server.py                      # Python Flask NLP microservice
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
pip install flask flask-cors sentence-transformers faiss-cpu spacy gensim numpy
python -m spacy download en_core_web_sm
```

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

### ML Intent Classification (Colab)

| Model | Accuracy | Macro AUC |
|---|---|---|
| Logistic Regression | 89.53% | >0.97 |
| SVM | 90.70% | >0.97 |
| **DistilBERT** | **93.60%** | **>0.97** |

Training data: 1001 samples, 8 classes

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
| /health | GET | Service status |
| /semantic-search | POST | Sentence-BERT + FAISS |
| /ner | POST | spaCy + custom NER |
| /similar-words | POST | Word2Vec similarity |
| /analyze | POST | Full NLP analysis |

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
- ML intent classification (DistilBERT 93.60%)
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
