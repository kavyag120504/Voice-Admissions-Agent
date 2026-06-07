/**
 * Python NLP Microservice Client
 * Connects to the Python Flask NLP service on port 5001
 * Provides: Sentence-BERT semantic search, spaCy NER, Word2Vec
 */

const NLP_SERVICE_URL = "http://localhost:5001";
let serviceAvailable = null; // null = not checked yet

async function checkAvailability() {
  try {
    const res = await fetch(`${NLP_SERVICE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    serviceAvailable = data.ok === true;
    if (serviceAvailable) {
      console.log(`[Python NLP] Service available — semantic:${data.semantic_search} ner:${data.spacy_ner} w2v:${data.word2vec}`);
    }
    return serviceAvailable;
  } catch {
    serviceAvailable = false;
    return false;
  }
}

// Check availability at startup
checkAvailability();

/**
 * Semantic search using Sentence-BERT + FAISS
 * Returns semantically similar knowledge base topics
 */
export async function semanticSearch(query, topK = 3) {
  if (serviceAvailable === false) return null;
  try {
    const res = await fetch(`${NLP_SERVICE_URL}/semantic-search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
      signal: AbortSignal.timeout(3000)
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.results?.length > 0 ? data : null;
  } catch {
    return null;
  }
}

/**
 * Named Entity Recognition using spaCy + custom BMU rules
 * Returns extracted entities from user query
 */
export async function extractEntitiesNLP(text) {
  if (serviceAvailable === false) return null;
  try {
    const res = await fetch(`${NLP_SERVICE_URL}/ner`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(2000)
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data;
  } catch {
    return null;
  }
}

/**
 * Full NLP analysis — combines NER + semantic search
 */
export async function analyzeQuery(query) {
  if (serviceAvailable === false) return null;
  try {
    const res = await fetch(`${NLP_SERVICE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: AbortSignal.timeout(3000)
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export const pythonNlpService = {
  isAvailable: () => serviceAvailable !== false,
  semanticSearch,
  extractEntitiesNLP,
  analyzeQuery,
  checkAvailability
};
