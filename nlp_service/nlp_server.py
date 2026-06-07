"""
BMU NLP Microservice
====================
Provides advanced NLP capabilities to the Node.js backend:
1. Semantic Search (Sentence-BERT + FAISS)
2. Named Entity Recognition (spaCy)
3. Word2Vec embeddings for query expansion
4. Intent classification (trained SVM/DistilBERT)

Runs on port 5001
"""

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
import json
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
if HAS_CORS:
    CORS(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "backend", "data")
FACTS_PATH = os.path.join(DATA_DIR, "bmu_facts.json")
DOCS_DIR = os.path.join(DATA_DIR, "..", "..", "..", "data", "documents")

# ── Load BMU knowledge base ───────────────────────────────────────────────────
def load_knowledge_base():
    try:
        with open(FACTS_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        docs = []
        for topic in kb.get("topics", []):
            docs.append({
                "id": topic["id"],
                "text": topic["answer_en"],
                "text_hi": topic.get("answer_hi", ""),
                "source": topic.get("source_title", "BMU"),
                "url": topic.get("source_url", "https://www.bmu.edu.in/")
            })
        logger.info(f"Loaded {len(docs)} topics from knowledge base")
        return docs
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        return []

KB_DOCS = load_knowledge_base()

# ── Sentence-BERT + FAISS Semantic Search ────────────────────────────────────
semantic_model = None
faiss_index = None
doc_embeddings = None

def init_semantic_search():
    global semantic_model, faiss_index, doc_embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        logger.info("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build FAISS index from knowledge base
        texts = [doc["text"] for doc in KB_DOCS]
        if not texts:
            logger.warning("No documents to index")
            return

        logger.info(f"Encoding {len(texts)} documents...")
        doc_embeddings = semantic_model.encode(texts, convert_to_numpy=True)
        doc_embeddings = doc_embeddings.astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / (norms + 1e-10)

        # Build FAISS index
        dim = doc_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity after normalization
        faiss_index.add(doc_embeddings)

        logger.info(f"FAISS index built with {faiss_index.ntotal} vectors (dim={dim})")
    except Exception as e:
        logger.error(f"Semantic search init failed: {e}")
        semantic_model = None

# ── spaCy NER ─────────────────────────────────────────────────────────────────
nlp_spacy = None

def init_spacy():
    global nlp_spacy
    try:
        import spacy
        # Try to load English model
        try:
            nlp_spacy = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            nlp_spacy = spacy.load("en_core_web_sm")
            logger.info("spaCy model downloaded and loaded")
    except Exception as e:
        logger.error(f"spaCy init failed: {e}")
        nlp_spacy = None

# ── Word2Vec ──────────────────────────────────────────────────────────────────
word2vec_model = None

def init_word2vec():
    global word2vec_model
    try:
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess

        # Build corpus from knowledge base
        corpus = []
        for doc in KB_DOCS:
            corpus.append(simple_preprocess(doc["text"]))
            if doc["text_hi"]:
                corpus.append(simple_preprocess(doc["text_hi"]))

        # Add domain vocabulary sentences
        domain_sentences = [
            "btech cse fees scholarship placement hostel admission",
            "computer science engineering program course branch",
            "scholarship merit discount waiver financial aid",
            "placement job salary package company recruiter",
            "hostel accommodation room food mess facility",
            "admission apply eligibility documents process",
            "fees tuition cost annual yearly semester",
            "location address gurugram delhi distance",
        ]
        for s in domain_sentences:
            corpus.append(simple_preprocess(s))

        if len(corpus) < 5:
            logger.warning("Corpus too small for Word2Vec")
            return

        word2vec_model = Word2Vec(
            sentences=corpus,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            epochs=50
        )
        logger.info(f"Word2Vec trained on {len(corpus)} sentences, vocab size: {len(word2vec_model.wv)}")
    except Exception as e:
        logger.error(f"Word2Vec init failed: {e}")
        word2vec_model = None

# ── Initialize all models ─────────────────────────────────────────────────────
logger.info("Initializing NLP models...")
init_semantic_search()
init_spacy()
init_word2vec()
logger.info("All models initialized")

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "semantic_search": semantic_model is not None,
        "spacy_ner": nlp_spacy is not None,
        "word2vec": word2vec_model is not None,
        "kb_docs": len(KB_DOCS)
    })

@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    """
    Semantic search using Sentence-BERT + FAISS
    Returns top-K most semantically similar knowledge base topics
    """
    if semantic_model is None or faiss_index is None:
        return jsonify({"error": "semantic_search_not_available", "results": []}), 503

    data = request.get_json()
    query = str(data.get("query", "")).strip()
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"results": [], "confidence": 0})

    try:
        # Encode query
        query_vec = semantic_model.encode([query], convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(query_vec)
        query_vec = query_vec / (norm + 1e-10)

        # Search FAISS
        scores, indices = faiss_index.search(query_vec, min(top_k, len(KB_DOCS)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < 0.2:
                continue
            doc = KB_DOCS[idx]
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "source": doc["source"],
                "url": doc["url"],
                "score": float(score)
            })

        confidence = float(results[0]["score"]) if results else 0.0

        return jsonify({
            "results": results,
            "confidence": round(confidence, 3),
            "query": query
        })
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return jsonify({"error": str(e), "results": []}), 500


@app.route("/ner", methods=["POST"])
def named_entity_recognition():
    """
    Named Entity Recognition using spaCy
    Extracts entities from user query
    """
    if nlp_spacy is None:
        return jsonify({"error": "spacy_not_available", "entities": []}), 503

    data = request.get_json()
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"entities": [], "text": ""})

    try:
        doc = nlp_spacy(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_) or ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Also extract BMU-specific entities with custom rules
        bmu_entities = extract_bmu_entities(text)

        return jsonify({
            "entities": entities,
            "bmu_entities": bmu_entities,
            "text": text
        })
    except Exception as e:
        logger.error(f"NER error: {e}")
        return jsonify({"error": str(e), "entities": []}), 500


def extract_bmu_entities(text):
    """Custom BMU domain entity extraction"""
    import re
    entities = {}

    # Programs
    programs = []
    if re.search(r'\bcse\b|computer science', text, re.I): programs.append("CSE")
    if re.search(r'\bmba\b', text, re.I): programs.append("MBA")
    if re.search(r'\bllb\b|law', text, re.I): programs.append("LLB")
    if re.search(r'\bmechanical\b|\bmech\b', text, re.I): programs.append("Mechanical")
    if re.search(r'\bbba\b', text, re.I): programs.append("BBA")
    if re.search(r'\become\b|electronics', text, re.I): programs.append("EComE")
    if re.search(r'\bbtech\b|b\.tech', text, re.I) and not programs: programs.append("BTech")
    if programs: entities["programs"] = programs

    # Percentage
    pct = re.search(r'(\d{2,3})\s*(?:%|percent|percentage|marks)', text, re.I)
    if pct: entities["percentage"] = int(pct.group(1))

    # Year
    yr = re.search(r'\b(1st|2nd|3rd|4th|first|second|third|fourth)\s*year\b', text, re.I)
    if yr: entities["year"] = yr.group(1)

    # Flags
    if re.search(r'\bhostel\b|accommodation|stay', text, re.I): entities["needs_hostel"] = True
    if re.search(r'\bscholarship\b|discount|waiver', text, re.I): entities["needs_scholarship"] = True

    return entities


@app.route("/word-similarity", methods=["POST"])
def word_similarity():
    """
    Word2Vec similarity between two terms
    Useful for query expansion
    """
    if word2vec_model is None:
        return jsonify({"error": "word2vec_not_available", "similarity": 0}), 503

    data = request.get_json()
    word1 = str(data.get("word1", "")).lower().strip()
    word2 = str(data.get("word2", "")).lower().strip()

    try:
        if word1 in word2vec_model.wv and word2 in word2vec_model.wv:
            sim = float(word2vec_model.wv.similarity(word1, word2))
            return jsonify({"similarity": round(sim, 4), "word1": word1, "word2": word2})
        else:
            missing = [w for w in [word1, word2] if w not in word2vec_model.wv]
            return jsonify({"similarity": 0, "missing_words": missing})
    except Exception as e:
        return jsonify({"error": str(e), "similarity": 0}), 500


@app.route("/similar-words", methods=["POST"])
def similar_words():
    """
    Get most similar words using Word2Vec
    Used for query expansion
    """
    if word2vec_model is None:
        return jsonify({"error": "word2vec_not_available", "similar": []}), 503

    data = request.get_json()
    word = str(data.get("word", "")).lower().strip()
    top_n = int(data.get("top_n", 5))

    try:
        if word in word2vec_model.wv:
            similar = word2vec_model.wv.most_similar(word, topn=top_n)
            return jsonify({
                "word": word,
                "similar": [{"word": w, "score": round(s, 4)} for w, s in similar]
            })
        else:
            return jsonify({"word": word, "similar": [], "error": "word_not_in_vocabulary"})
    except Exception as e:
        return jsonify({"error": str(e), "similar": []}), 500


@app.route("/analyze", methods=["POST"])
def full_analysis():
    """
    Full NLP analysis of a query:
    - NER (spaCy + custom BMU)
    - Semantic search
    - Word2Vec similar terms for top keywords
    """
    data = request.get_json()
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "empty_query"}), 400

    result = {"query": query}

    # NER
    if nlp_spacy:
        try:
            doc = nlp_spacy(query)
            result["spacy_entities"] = [
                {"text": e.text, "label": e.label_} for e in doc.ents
            ]
            result["bmu_entities"] = extract_bmu_entities(query)
        except Exception as e:
            result["ner_error"] = str(e)

    # Semantic search
    if semantic_model and faiss_index:
        try:
            query_vec = semantic_model.encode([query], convert_to_numpy=True).astype(np.float32)
            norm = np.linalg.norm(query_vec)
            query_vec = query_vec / (norm + 1e-10)
            scores, indices = faiss_index.search(query_vec, 3)
            result["semantic_matches"] = [
                {
                    "id": KB_DOCS[idx]["id"],
                    "score": round(float(score), 3)
                }
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0 and float(score) > 0.2
            ]
        except Exception as e:
            result["semantic_error"] = str(e)

    return jsonify(result)


if __name__ == "__main__":
    import spacy
    logger.info("Starting BMU NLP Microservice on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=False)
