import { distance } from "fastest-levenshtein";

// BMU domain-specific vocabulary — only words that students might misspell
// Deliberately excludes common English words to avoid false corrections
const BMU_VOCAB = [
  "btech","mtech","mba","bba","llb","phd","bcom","cse","ece","ecome",
  "mechanical","engineering","management","law","commerce","computer","science",
  "electronics","automobile","robotics","cybersecurity","artificial","intelligence",
  "admission","admissions","eligibility","scholarship","scholarships","placement",
  "placements","hostel","cafeteria","gurugram","gurgaon","sidhrawali","munjal",
  "university","college","institute","naac","nirf","tuition","semester","curriculum",
  "specialization","internship","recruitment","counselling","brochure","marksheet",
  "certificate","aadhar","clat","cuet","liberal","integrated","executive"
];

// Common English + Hinglish words that should NEVER be spell-corrected
// (they are valid as-is and short words get corrupted easily)
const NEVER_CORRECT = new Set([
  "how","far","is","bmu","from","where","what","when","who","which","why",
  "the","and","or","in","on","at","to","for","of","with","by","as","an","a",
  "hi","hello","hey","ok","okay","yes","no","not","can","do","does","did",
  "hai","hain","ka","ki","ke","mein","se","ko","kya","aur","ya","bhi","hi",
  "to","na","nahi","tha","thi","the","aap","main","hum","woh","yeh","ye",
  "kab","kahan","kaun","kaunsi","kaise","kitna","kitni","kitne","kyun",
  "batao","bata","please","sir","mam","ji","ek","do","teen","char",
  "tell","me","my","us","our","you","your","he","she","it","they","them",
  "this","that","these","those","will","would","could","should","may","might",
  "have","has","had","been","being","are","was","were","be","am",
  "also","just","very","too","more","most","some","any","all","both",
  "about","above","after","before","between","into","through","during",
  "fee","fees","job","jobs","law","arts","far","near","good","best",
  "btech","mtech","mba","bba","llb","cse","ece","bcom","phd","bmu","bml",
  "clat","jee","cat","mat","sat","cuet","gate","naac","nirf","lpa","ctc"
]);

const correctionCache = new Map();

function correctToken(token) {
  const t = token.toLowerCase();

  // Never correct short words, stopwords, or known-good words
  if (t.length <= 3) return token;
  if (NEVER_CORRECT.has(t)) return token;
  if (correctionCache.has(t)) return correctionCache.get(t);

  let bestMatch = null;
  let bestDist = Infinity;

  for (const word of BMU_VOCAB) {
    if (Math.abs(word.length - t.length) > 3) continue;
    const d = distance(t, word);
    // Only correct if edit distance is 1-2 AND the word is meaningfully different
    if (d > 0 && d <= 2 && d < bestDist) {
      bestDist = d;
      bestMatch = word;
    }
  }

  // Only apply correction if confident (distance 1 always, distance 2 only for longer words)
  const result = (bestMatch && (bestDist === 1 || (bestDist === 2 && t.length >= 7)))
    ? bestMatch
    : token;

  correctionCache.set(t, result);
  return result;
}

export function spellCorrect(text) {
  const tokens = String(text || "").split(/\s+/).filter(Boolean);
  const corrections = [];

  const corrected = tokens.map(token => {
    // Preserve punctuation
    const clean = token.replace(/[^a-zA-Z0-9\u0900-\u097F]/g, "");
    if (!clean || /[\u0900-\u097F]/.test(clean)) return token; // skip Devanagari
    const fixed = correctToken(clean);
    if (fixed !== clean) corrections.push({ original: clean, corrected: fixed });
    return token.replace(clean, fixed);
  });

  return {
    text: corrected.join(" "),
    corrections,
    wasChanged: corrections.length > 0
  };
}
