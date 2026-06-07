import { config } from "../config.js";

function splitSentences(text) {
  return String(text || "")
    .split(/(?<=[.!?।])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function clampSentences(text, maxSentences = config.maxResponseSentences) {
  const parts = splitSentences(text);
  if (parts.length <= maxSentences) return String(text || "").trim();
  return `${parts.slice(0, maxSentences).join(" ").trim()}`;
}

function removeOverpromises(text) {
  return String(text || "")
    .replace(/\b(guaranteed|100%|always accurate|perfectly accurate)\b/gi, "reliable")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export const responseQualityService = {
  enforce(answerText, { retrievalConfidence = 0, isHindiMode = false, strictMode = config.strictDemoMode } = {}) {
    let result = removeOverpromises(answerText);
    result = clampSentences(result);

    if (strictMode && retrievalConfidence < config.minRetrievalConfidence) {
      return isHindiMode
        ? "सटीक जानकारी देने के लिए एक बिंदु स्पष्ट करें: program, year, और requirement बताइए, फिर मैं verified जवाब दूँगा।"
        : "To give an accurate verified answer, please confirm one point: program, year, and requirement.";
    }

    return result;
  }
};
