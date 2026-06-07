import natural from "natural";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "../config.js";
import { preprocessQuery } from "./nlpService.js";

const { TfIdf, PorterStemmer } = natural;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const localDataDir = path.resolve(__dirname, "../../data");

function safeRead(filePath) {
  try { return fs.readFileSync(filePath, "utf-8"); } catch { return ""; }
}

function cleanText(raw) {
  // Strip HTML tags and normalize whitespace
  return String(raw || "")
    .replace(/<[^>]+>/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function chunkText(raw, source) {
  return cleanText(raw)
    .split(/\n{2,}|\.\s{2,}/)
    .map(b => b.replace(/\s+/g, " ").trim())
    .filter(b => b.length > 80)
    .map((text, idx) => ({ text, source, chunkId: `${source}#${idx + 1}` }));
}

// Build corpus from all data sources
function loadCorpus() {
  const sources = [];
  const docDir = path.join(config.globalDataDir, "documents");
  const scrapedDir = path.join(config.globalDataDir, "scraped");
  const localFacts = path.join(localDataDir, "bmu_facts.json");

  if (fs.existsSync(docDir)) {
    for (const name of fs.readdirSync(docDir)) {
      if (!name.endsWith(".txt")) continue;
      sources.push({ source: `documents/${name}`, content: safeRead(path.join(docDir, name)) });
    }
  }

  if (fs.existsSync(scrapedDir)) {
    for (const name of fs.readdirSync(scrapedDir)) {
      if (!name.endsWith("_content.txt")) continue;
      sources.push({ source: `scraped/${name}`, content: safeRead(path.join(scrapedDir, name)) });
    }
  }

  // Also add bmu_facts answers as documents
  if (fs.existsSync(localFacts)) {
    try {
      const facts = JSON.parse(safeRead(localFacts));
      for (const topic of facts.topics || []) {
        sources.push({
          source: `facts/${topic.id}`,
          content: `${topic.answer_en} ${topic.answer_hi || ""}`
        });
      }
    } catch { /* ignore */ }
  }

  const chunks = [];
  for (const src of sources) {
    chunks.push(...chunkText(src.content, src.source));
  }
  return chunks;
}

// Build TF-IDF index at startup
const corpusChunks = loadCorpus();
const tfidf = new TfIdf();

for (const chunk of corpusChunks) {
  // Stem each word before adding to index for better matching
  const stemmed = chunk.text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(w => w.length > 2)
    .map(w => PorterStemmer.stem(w))
    .join(" ");
  tfidf.addDocument(stemmed);
}

console.log(`[TF-IDF] Indexed ${corpusChunks.length} chunks from ${new Set(corpusChunks.map(c => c.source)).size} sources`);

/**
 * Retrieve top-K most relevant chunks for a query using proper TF-IDF scoring
 */
export function tfidfRetrieve(query, topK = config.retrievalTopK) {
  const { concepts, stems } = preprocessQuery(query);

  // Build stemmed query string for TF-IDF
  const queryTerms = [...new Set([...concepts, ...stems])]
    .map(t => PorterStemmer.stem(t))
    .join(" ");

  if (!queryTerms.trim()) return { snippets: [], confidence: 0 };

  // Score all documents
  const scores = [];
  tfidf.tfidfs(queryTerms, (i, measure) => {
    if (measure > 0) {
      scores.push({ index: i, score: measure });
    }
  });

  // Sort by score descending
  scores.sort((a, b) => b.score - a.score);

  // Deduplicate by source — max 1 chunk per source
  const seenSources = new Set();
  const topChunks = [];
  for (const { index, score } of scores) {
    if (topChunks.length >= topK) break;
    const chunk = corpusChunks[index];
    if (!chunk) continue;
    if (seenSources.has(chunk.source)) continue;
    seenSources.add(chunk.source);
    topChunks.push({ ...chunk, score });
  }

  const avgScore = topChunks.length
    ? topChunks.reduce((s, c) => s + c.score, 0) / topChunks.length
    : 0;

  // Normalize confidence to 0-1 range
  const confidence = Number(Math.min(0.95, avgScore / 20).toFixed(2));

  return { snippets: topChunks, confidence };
}
