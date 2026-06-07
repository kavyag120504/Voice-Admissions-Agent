const replacements = [
  [/kol(d|्ड)?/gi, "could"],
  [/cors|कोर्स|course/gi, "course"],
  [/bihavior|behavior|behaviour/gi, "behavior"],
  [/ofers|offers|ऑफर्स/gi, "offers"],
  [/plij|please|प्लीज/gi, "please"]
];

function normalizeWhitespace(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function repairText(text) {
  let out = normalizeWhitespace(text);
  for (const [pattern, replaceWith] of replacements) {
    out = out.replace(pattern, replaceWith);
  }
  return out;
}

export const asrFallbackService = {
  repairTranscript(inputText = "", confidence = 1) {
    const clean = normalizeWhitespace(inputText);
    if (!clean) return { text: "", usedFallback: false };

    const lowConfidence = Number(confidence) < 0.75;
    if (!lowConfidence) {
      return { text: clean, usedFallback: false };
    }
    return { text: repairText(clean), usedFallback: true };
  }
};
