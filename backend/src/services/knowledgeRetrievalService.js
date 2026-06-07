import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "../config.js";
import { preprocessQuery } from "./nlpService.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const localDataDir = path.resolve(__dirname, "../../data");

function safeRead(filePath) {
  try {
    return fs.readFileSync(filePath, "utf-8");
  } catch {
    return "";
  }
}

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

function buildQueryTerms(query) {
  // Use NLP pipeline — stemming + concept mapping handles plurals, verb forms, Hinglish
  const { concepts, stems } = preprocessQuery(query);
  const terms = new Set([...concepts, ...stems]);
  return terms;
}

function chunkText(raw, source) {
  const blocks = String(raw || "")
    .split(/\n{2,}/)
    .map((b) => b.replace(/\s+/g, " ").trim())
    .filter((b) => b.length > 80);
  return blocks.map((text, idx) => ({ text, source, chunkId: `${source}#${idx + 1}` }));
}

function scoreChunk(chunk, queryTerms) {
  const words = new Set(tokenize(chunk.text));
  let overlap = 0;
  for (const term of queryTerms) {
    if (words.has(term)) overlap += 1;
  }
  const lengthBonus = Math.min(0.4, chunk.text.length / 2500);
  return overlap + lengthBonus;
}

function normalizeKey(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 220);
}

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

  if (fs.existsSync(localFacts)) {
    const json = safeRead(localFacts);
    sources.push({ source: "local/bmu_facts.json", content: json });
  }

  const chunks = [];
  for (const src of sources) {
    chunks.push(...chunkText(src.content, src.source));
  }
  return chunks;
}

const corpusChunks = loadCorpus();

export const knowledgeRetrievalService = {
  retrieve(query, topK = config.retrievalTopK) {
    const queryTerms = buildQueryTerms(query);
    const scoredRaw = corpusChunks
      .map((chunk) => ({ ...chunk, score: scoreChunk(chunk, queryTerms) }))
      .filter((c) => c.score >= 1.2)
      .sort((a, b) => b.score - a.score);

    const seenText = new Set();
    const sourceUsage = new Map();
    const scored = [];
    for (const candidate of scoredRaw) {
      if (scored.length >= topK) break;
      const key = normalizeKey(candidate.text);
      if (!key || seenText.has(key)) continue;
      const sourceCount = sourceUsage.get(candidate.source) || 0;
      if (sourceCount >= 1) continue;
      seenText.add(key);
      sourceUsage.set(candidate.source, sourceCount + 1);
      scored.push(candidate);
    }

    const averageScore = scored.length
      ? scored.reduce((acc, c) => acc + c.score, 0) / scored.length
      : 0;

    return {
      snippets: scored,
      confidence: Number(Math.min(0.95, averageScore / 5).toFixed(2))
    };
  }
};
