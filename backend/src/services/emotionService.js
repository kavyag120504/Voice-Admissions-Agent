const emotionLexicon = {
  stressed: ["stress", "worried", "tension", "panic", "confused", "nervous", "anxious", "pressure"],
  sad: ["sad", "upset", "low", "depressed", "hopeless", "down"],
  happy: ["happy", "great", "awesome", "excellent", "excited", "relieved"],
  angry: ["angry", "frustrated", "irritated", "annoyed", "furious"],
  neutral: []
};

function scoreEmotion(text) {
  const normalized = (text || "").toLowerCase();
  const sanitized = normalized.replace(/happy to help|be happy to help|glad to help/g, "");
  const scores = {
    stressed: 0,
    sad: 0,
    happy: 0,
    angry: 0,
    neutral: 1
  };

  Object.entries(emotionLexicon).forEach(([emotion, words]) => {
    words.forEach((word) => {
      if (sanitized.includes(word)) scores[emotion] += 1;
    });
  });

  if (normalized.includes("!")) scores.happy += 0.5;
  if (normalized.includes("?")) scores.stressed += 0.25;

  let topEmotion = "neutral";
  let maxScore = -Infinity;
  for (const [emotion, value] of Object.entries(scores)) {
    if (value > maxScore) {
      maxScore = value;
      topEmotion = emotion;
    }
  }

  const confidence = Math.min(0.95, 0.3 + maxScore * 0.18);
  return { emotion: topEmotion, confidence: Number(confidence.toFixed(2)), scores };
}

function detectLanguageMode(text) {
  const input = text || "";
  const hasDevanagari = /[\u0900-\u097F]/.test(input);
  const hasLatin = /[A-Za-z]/.test(input);

  if (hasDevanagari && hasLatin) return "hinglish";
  if (hasDevanagari) return "hindi";

  // Detect Roman Hinglish — common Hindi words written in English
  const hinglishWords = /\b(kya|hai|hain|kaise|kitna|kitni|kahan|kaun|kaunsi|mein|se|ko|ka|ki|ke|aur|ya|bhi|nahi|nhi|hoga|milega|milegi|chahiye|batao|bata|aap|main|hum|toh|agar|lekin|par|pe|woh|yeh|ye|ek|do|teen|char|sab|kuch|bahut|thoda|zyada|accha|theek|sahi|galat|lakh|crore|rupee|rupees|paisa|paise)\b/i;

  if (hinglishWords.test(input)) return "hinglish";

  return "english";
}

export const emotionService = {
  analyzeUserSignal(text) {
    return {
      ...scoreEmotion(text),
      languageMode: detectLanguageMode(text)
    };
  }
};
