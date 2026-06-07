import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { preprocessQuery, scoreTopicWithIntent, detectIntent } from "./nlpService.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const factsPath = path.join(__dirname, "../../data/bmu_facts.json");

let knowledgeBase = { topics: [] };
try {
  knowledgeBase = JSON.parse(fs.readFileSync(factsPath, "utf-8"));
} catch {
  knowledgeBase = { topics: [] };
}

function containsDevanagari(text) {
  return /[\u0900-\u097F]/.test(text || "");
}

function findTopic(userText) {
  const queryResult = preprocessQuery(userText);
  const intents = detectIntent(userText);
  let best = null;
  let bestScore = 0;

  for (const topic of knowledgeBase.topics || []) {
    const score = scoreTopicWithIntent(topic.keywords, topic.id, queryResult, intents);
    if (score > bestScore) {
      bestScore = score;
      best = topic;
    }
  }

  return bestScore >= 1 ? { topic: best, score: bestScore } : null;
}

function toConfidence(score) {
  const confidence = Math.min(0.96, 0.55 + score * 0.1);
  return Number(confidence.toFixed(2));
}

export const responsePolicy = {
  resolveGroundedAnswer(userText, languageMode) {
    const match = findTopic(userText);
    if (!match) return null;
    const preferHindi = languageMode === "hindi" || languageMode === "hinglish" || containsDevanagari(userText);
    return {
      text: preferHindi ? match.topic.answer_hi : match.topic.answer_en,
      source: {
        title: match.topic.source_title || "BMU Official Source",
        url: match.topic.source_url || "https://www.bmu.edu.in/",
        topicId: match.topic.id
      },
      confidence: toConfidence(match.score)
    };
  }
};
