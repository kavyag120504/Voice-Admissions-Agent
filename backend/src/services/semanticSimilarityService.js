import natural from "natural";
const { TfIdf, PorterStemmer } = natural;

/**
 * Semantic Similarity Service
 * Uses TF-IDF cosine similarity to detect paraphrases and similar queries
 * This approximates sentence embedding similarity without needing BERT/GPU
 *
 * Example:
 *   "job opportunities at BMU" ≈ "placement statistics"
 *   "cost of studying" ≈ "fees"
 *   "where is the university" ≈ "location of BMU"
 */

// Pre-defined paraphrase clusters — queries that mean the same thing
const PARAPHRASE_CLUSTERS = [
  {
    canonical: "btech_fees",
    phrases: [
      "what is the fee", "how much does it cost", "cost of studying",
      "tuition charges", "how much is btech", "fees for engineering",
      "kitna paisa lagega", "fees kitni hai", "kharcha kitna hai"
    ]
  },
  {
    canonical: "placements",
    phrases: [
      "job opportunities", "career prospects", "employment after bmu",
      "companies visiting", "campus recruitment", "salary after graduation",
      "naukri milegi", "job milega", "placement ka scene"
    ]
  },
  {
    canonical: "scholarships",
    phrases: [
      "fee reduction", "financial support", "merit benefits",
      "fee discount", "can i get help with fees", "fee waiver",
      "fees kam hogi", "koi discount hai", "financial help"
    ]
  },
  {
    canonical: "admission_process",
    phrases: [
      "how to join bmu", "enrollment process", "how to get admission",
      "steps to apply", "application procedure", "joining process",
      "kaise join karein", "admission kaise hoga", "apply kaise karein"
    ]
  },
  {
    canonical: "contact_info",
    phrases: [
      "how to reach bmu", "bmu address", "where is the campus",
      "directions to bmu", "bmu location", "how far is bmu",
      "bmu kahan hai", "campus kahan hai", "address kya hai"
    ]
  },
  {
    canonical: "hostel",
    phrases: [
      "accommodation options", "where to stay", "living on campus",
      "room and board", "residential facilities", "stay at bmu",
      "rehne ki jagah", "hostel ki suvidha", "campus mein rehna"
    ]
  }
];

// Build TF-IDF index for similarity computation
const tfidf = new TfIdf();
const clusterIndex = []; // maps document index → canonical topic

for (const cluster of PARAPHRASE_CLUSTERS) {
  for (const phrase of cluster.phrases) {
    const stemmed = phrase.toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .split(/\s+/)
      .filter(w => w.length > 1)
      .map(w => PorterStemmer.stem(w))
      .join(" ");
    tfidf.addDocument(stemmed);
    clusterIndex.push(cluster.canonical);
  }
}

/**
 * Compute cosine similarity between two TF-IDF vectors
 */
function cosineSimilarity(vec1, vec2) {
  const keys = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
  let dot = 0, mag1 = 0, mag2 = 0;
  for (const k of keys) {
    const a = vec1[k] || 0;
    const b = vec2[k] || 0;
    dot += a * b;
    mag1 += a * a;
    mag2 += b * b;
  }
  if (!mag1 || !mag2) return 0;
  return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
}

/**
 * Get TF-IDF vector for a text
 */
function getVector(text) {
  const stemmed = text.toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(w => w.length > 1)
    .map(w => PorterStemmer.stem(w))
    .join(" ");

  const tempTfidf = new TfIdf();
  tempTfidf.addDocument(stemmed);
  const vec = {};
  tempTfidf.listTerms(0).forEach(item => { vec[item.term] = item.tfidf; });
  return vec;
}

/**
 * Find the most semantically similar canonical topic for a query
 * Returns null if similarity is below threshold
 */
export function findSimilarTopic(query, threshold = 0.25) {
  const queryVec = getVector(query);
  if (Object.keys(queryVec).length === 0) return null;

  let bestScore = 0;
  let bestCanonical = null;

  tfidf.documents.forEach((doc, idx) => {
    const docVec = {};
    Object.keys(doc).forEach(term => {
      if (term !== "__key") docVec[term] = tfidf.tfidf(term, idx);
    });
    const score = cosineSimilarity(queryVec, docVec);
    if (score > bestScore) {
      bestScore = score;
      bestCanonical = clusterIndex[idx];
    }
  });

  return bestScore >= threshold ? { canonical: bestCanonical, score: bestScore } : null;
}

/**
 * Check if two queries are paraphrases of each other
 */
export function areParaphrases(query1, query2, threshold = 0.6) {
  const vec1 = getVector(query1);
  const vec2 = getVector(query2);
  return cosineSimilarity(vec1, vec2) >= threshold;
}
