import natural from "natural";
import { distance } from "fastest-levenshtein";

const { PorterStemmer, SoundEx } = natural;
const soundex = new SoundEx();

// ─── STOPWORDS ────────────────────────────────────────────────────────────────

const STOPWORDS = new Set([
  // English function words
  "a","an","the","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","could","should","may","might","shall","can",
  "i","me","my","we","our","you","your","he","she","it","they","them","their",
  "this","that","these","those","am","at","by","for","in","of","on","to","up",
  "as","or","and","but","if","so","yet","both","each","about","above","after",
  "before","between","into","through","during","with","from","all","any","both",
  "few","more","most","other","some","such","no","not","only","same","than",
  "too","very","just","because","while","also","then","than","its","it's",
  // Hinglish function words (NOT content words like kahan, kaunsi)
  "hai","hain","ka","ki","ke","mein","se","ko","ne","par","aur","ya",
  "bhi","hi","to","na","nahi","nhi","tha","thi","the","hoga","hogi","honge",
  "kyun","kyunki","mujhe","muje","humko","humein","aapko","aap","main","hum",
  "woh","yeh","ye","ek","do","teen","char","please","plz","sir","mam","bhai",
  "ji","okay","ok","hota","hoti","batao","bata","chahie","milega","milegi",
  "milenge","lagta","lagti","chahiye"
]);

// ─── CONCEPT MAP ─────────────────────────────────────────────────────────────
// Maps any surface form → normalized concept used for topic matching

const CONCEPT_MAP = {
  // === LOCATION / CONTACT ===
  "where": "contact", "wher": "contact", "whre": "contact",
  "kahan": "contact", "kha": "contact", "kidhar": "contact",
  "location": "contact", "address": "contact", "situated": "contact",
  "located": "contact", "distance": "contact", "far": "contact",
  "reach": "contact", "direction": "contact", "how far": "contact",
  "kitni door": "contact", "door": "contact", "paas": "contact",
  "delhi": "contact", "gurugram": "contact", "gurgaon": "contact",
  "haryana": "contact", "nh48": "contact", "highway": "contact",
  "sidhrawali": "contact", "phone": "contact", "email": "contact",
  "contact": "contact", "number": "contact",

  // === FEES ===
  "fee": "fee", "fees": "fee", "fess": "fee", "fies": "fee", "fea": "fee",
  "tuition": "fee", "cost": "fee", "price": "fee", "charge": "fee",
  "charges": "fee", "paisa": "fee", "paise": "fee", "rupee": "fee",
  "rupees": "fee", "kharcha": "fee", "kharch": "fee", "kitna": "fee",
  "kitni": "fee", "payment": "fee", "pay": "fee", "total": "fee",
  "overall": "fee", "annual": "fee", "yearly": "fee",

  // === SCHOLARSHIP ===
  "scholarship": "scholarship", "scholarships": "scholarship",
  "scholership": "scholarship", "scolarship": "scholarship",
  "skolarship": "scholarship", "scholrship": "scholarship",
  "waiver": "scholarship", "discount": "scholarship",
  "concession": "scholarship", "chhutt": "scholarship",
  "chhoot": "scholarship", "free": "scholarship", "aid": "scholarship",
  "financial": "scholarship", "merit": "scholarship",

  // === ADMISSION ===
  "admission": "admission", "admissions": "admission",
  "addmission": "admission", "admision": "admission",
  "addmision": "admission", "admsn": "admission",
  "apply": "admission", "applying": "admission", "applied": "admission",
  "application": "admission", "form": "admission",
  "register": "admission", "registration": "admission",
  "enroll": "admission", "enrollment": "admission",
  "lena": "admission", "leni": "admission", "join": "admission",

  // === ELIGIBILITY ===
  "eligibility": "eligibility", "eligible": "eligibility",
  "eligiblity": "eligibility", "eligibilty": "eligibility",
  "criteria": "eligibility", "criterion": "eligibility",
  "marks": "eligibility", "percentage": "eligibility",
  "percent": "eligibility", "score": "eligibility",
  "qualify": "eligibility", "qualified": "eligibility",
  "required": "eligibility", "requirement": "eligibility",
  "minimum": "eligibility", "cutoff": "eligibility",

  // === DOCUMENTS ===
  "document": "documents", "documents": "documents",
  "docs": "documents", "certificate": "documents",
  "marksheet": "documents", "marksheets": "documents",
  "aadhar": "documents", "transfer": "documents",
  "character": "documents", "papers": "documents",
  "need": "documents", "required": "documents", "requirement": "documents",

  // === HOSTEL ===
  "hostel": "hostel", "hostels": "hostel", "hostle": "hostel",
  "hostl": "hostel", "accommodation": "hostel", "room": "hostel",
  "rooms": "hostel", "stay": "hostel", "rehna": "hostel",
  "reh": "hostel", "mess": "hostel", "food": "hostel",
  "canteen": "hostel", "pg": "hostel", "khana": "hostel",
  "khaana": "hostel", "meal": "hostel", "meals": "hostel",
  "dining": "hostel", "lunch": "hostel", "dinner": "hostel",
  "breakfast": "hostel",

  // === PLACEMENT ===
  "placement": "placement", "placements": "placement",
  "placment": "placement", "plcement": "placement",
  "job": "placement", "jobs": "placement", "naukri": "placement",
  "salary": "placement", "salaries": "placement",
  "package": "placement", "packages": "placement",
  "ctc": "placement", "lpa": "placement", "lakh": "placement",
  "recruit": "placement", "recruiter": "placement",
  "recruiters": "placement", "company": "placement",
  "companies": "placement", "companie": "placement",
  "compny": "placement", "hiring": "placement", "hire": "placement",
  "placed": "placement", "campus": "placement",
  "kaunsi": "placement", "kaun": "placement", "aati": "placement",
  "scene": "placement", "career": "placement", "careers": "placement",
  "prospect": "placement", "prospects": "placement",
  "opportunity": "placement", "opportunities": "placement",
  "employment": "placement", "employed": "placement",

  // === COURSES ===
  "course": "course", "courses": "course", "korse": "course",
  "coarse": "course", "corse": "course",
  "program": "course", "programmes": "course", "programs": "course",
  "branch": "course", "branches": "course", "stream": "course",
  "streams": "course", "kaunse": "course",
  "offer": "course", "offered": "course", "padhai": "course",
  "curriculum": "course", "syllabus": "course",
  "all": "course",
  // === BTECH ===
  "btech": "btech", "b.tech": "btech", "btech.": "btech",
  "btech,": "btech", "btec": "btech", "b tech": "btech",
  "engineering": "btech", "engg": "btech", "eng": "btech",
  "engineer": "btech",

  // === CSE ===
  "cse": "cse", "cs": "cse", "computer": "cse", "computers": "cse",
  "coding": "cse", "software": "cse", "programming": "cse",
  "it": "cse", "information technology": "cse",
  "specialization": "cse", "specializations": "cse",

  // === MECHANICAL ===
  "mechanical": "mechanical", "mech": "mechanical",
  "automobile": "mechanical", "auto": "mechanical",

  // === ELECTRONICS ===
  "electronics": "electronics", "ecome": "electronics",
  "ece": "electronics", "electrical": "electronics",
  "vlsi": "electronics", "embedded": "electronics",

  // === MBA ===
  "mba": "mba", "management": "mba", "business": "mba",
  "manager": "mba", "executive": "mba",

  // === LAW ===
  "law": "law", "llb": "law", "legal": "law", "lawyer": "law",
  "advocate": "law", "barrister": "law", "clat": "law",
  "litigation": "law", "corporate": "law",

  // === CAMPUS / FACILITIES ===
  "library": "campus", "lab": "campus", "labs": "campus",
  "gym": "campus", "sports": "campus", "cafeteria": "campus",
  "medical": "campus", "wifi": "campus", "security": "campus",
  "transport": "campus", "facilities": "campus", "facility": "campus",
  "infrastructure": "campus", "mess": "campus", "canteen": "campus",
  "amenities": "campus", "equipment": "campus", "classroom": "campus",
  "auditorium": "campus", "ground": "campus", "court": "campus",

  // === BMU GENERAL ===
  "bmu": "bmu", "bml": "bmu", "munjal": "bmu",
  "university": "bmu", "college": "bmu", "institute": "bmu",
  "naac": "bmu", "nirf": "bmu", "qs": "bmu", "ranking": "bmu",
  "ranked": "bmu", "accredited": "bmu", "hero": "bmu",

  // === ABOUT ===
  "about": "about", "tell": "about", "explain": "about",
  "describe": "about", "overview": "about", "info": "about",
  "information": "about", "details": "about", "detail": "about",
  "batao": "about", "bataiye": "about",
  // === ACCREDITATION ===
  "ugc": "accreditation", "recognized": "accreditation", "accredited": "accreditation",
  "accreditation": "accreditation", "recognition": "accreditation",
  "approved": "accreditation", "legitimate": "accreditation", "valid": "accreditation",
  "aicte": "accreditation", "deemed": "accreditation", "autonomous": "accreditation",
  // === WHY BMU ===
  "unique": "why_bmu", "different": "why_bmu", "special": "why_bmu",
  "worth": "why_bmu", "justified": "why_bmu", "investment": "why_bmu",
  "alag": "why_bmu", "khasiyat": "why_bmu", "better": "why_bmu",
  "choose": "why_bmu", "expensive": "why_bmu", "costly": "why_bmu",
  "zyada": "why_bmu", "premium": "why_bmu", "costly": "why_bmu",
  // === SAFETY ===
  "safe": "safety", "safety": "safety", "secure": "safety",
  "security": "safety", "cctv": "safety", "surveillance": "safety",
  "attendance": "safety", "guard": "safety", "protection": "safety",
  // === EXTRACURRICULAR ===
  "extracurricular": "extracurricular", "clubs": "extracurricular",
  "fest": "extracurricular", "cultural": "extracurricular",
  "exchange": "extracurricular", "international": "extracurricular",
  "global": "extracurricular", "hackathon": "extracurricular",
  "society": "extracurricular", "societies": "extracurricular",
  "student life": "extracurricular", "life": "extracurricular",
  "activities": "extracurricular", "activity": "extracurricular",
  // === INTERNSHIP ===
  "internship": "internship", "internships": "internship",
  "industry": "internship", "collaboration": "internship",
  "training": "internship", "exposure": "internship",
  "stipend": "internship", "mentor": "internship", "mentorship": "internship",
  "hero group": "internship", "hero": "internship",

  // === NULL MAPPINGS — words that should NOT map to any concept ===
  // These prevent fuzzy/phonetic matching from producing wrong concepts
  "milta": null, "milti": null, "milte": null,
  "liye": null,
  "wala": null, "wali": null, "wale": null,
  "kaisa": null, "kaisi": null, "kaise": null,
  "lagega": null, "lagegi": null, "lagenge": null,
  "interest": null, "chahta": null, "chahti": null,
  "mujhe": null, "humko": null, "aapko": null,
  "exactly": null, "basically": null, "actually": null,
  "really": null, "totally": null, "generally": null,
  // Common English verbs/words that should never map to domain concepts
  "want": null, "know": null, "need": null,
  "like": null, "give": null, "show": null, "find": null,
  "tell": null, "said": null, "says": null, "saying": null,
  "good": null, "best": null, "great": null, "nice": null,
  "please": null, "help": null, "more": null, "also": null,
  "information": null, "details": null, "info": null
};

// Pre-build SoundEx codes for all concept map keys for phonetic matching
const SOUNDEX_MAP = new Map();
for (const key of Object.keys(CONCEPT_MAP)) {
  if (key.length > 2 && /^[a-z]+$/.test(key)) {
    const code = soundex.process(key);
    if (!SOUNDEX_MAP.has(code)) SOUNDEX_MAP.set(code, []);
    SOUNDEX_MAP.get(code).push(key);
  }
}

// Pre-build stem → concept map
const STEM_TO_CONCEPT = new Map();
for (const [word, concept] of Object.entries(CONCEPT_MAP)) {
  if (/^[a-z]+$/.test(word)) {
    const stem = PorterStemmer.stem(word);
    if (!STEM_TO_CONCEPT.has(stem)) STEM_TO_CONCEPT.set(stem, concept);
  }
}

// ─── OOV RESOLVER ────────────────────────────────────────────────────────────

const oovCache = new Map();

/**
 * Resolve an out-of-vocabulary token to a concept using:
 * 1. Direct map lookup
 * 2. Stem lookup
 * 3. Phonetic (SoundEx) matching
 * 4. Fuzzy (Levenshtein) matching — last resort
 */
function resolveToken(token) {
  if (oovCache.has(token)) return oovCache.get(token);

  // 1. Direct concept map (including null mappings)
  if (token in CONCEPT_MAP) {
    const val = CONCEPT_MAP[token];
    oovCache.set(token, val);
    return val; // may be null — caller handles it
  }

  // 2. Stem → concept
  const stem = PorterStemmer.stem(token);
  if (STEM_TO_CONCEPT.has(stem)) {
    const concept = STEM_TO_CONCEPT.get(stem);
    oovCache.set(token, concept);
    return concept;
  }

  // 3. Phonetic matching (SoundEx) — catches "scolarship" → "scholarship"
  if (token.length > 3 && /^[a-z]+$/.test(token)) {
    const code = soundex.process(token);
    const phonMatches = SOUNDEX_MAP.get(code) || [];
    if (phonMatches.length > 0) {
      // Pick closest by edit distance among phonetic matches
      let best = null, bestDist = Infinity;
      for (const candidate of phonMatches) {
        const d = distance(token, candidate);
        if (d < bestDist) { bestDist = d; best = candidate; }
      }
      if (best && bestDist <= 3) {
        const concept = CONCEPT_MAP[best];
        oovCache.set(token, concept);
        return concept;
      }
    }

    // 4. Fuzzy Levenshtein fallback — only for tokens >= 6 chars to avoid false matches
    if (token.length >= 6) {
      let bestWord = null, bestD = Infinity;
      for (const word of Object.keys(CONCEPT_MAP)) {
        if (Math.abs(word.length - token.length) > 3) continue;
        if (!/^[a-z]+$/.test(word)) continue;
        const d = distance(token, word);
        if (d < bestD && d <= 2) { bestD = d; bestWord = word; }
      }
      if (bestWord) {
        const concept = CONCEPT_MAP[bestWord];
        oovCache.set(token, concept);
        return concept;
      }
    }
  }

  // No match — keep original token and its stem
  oovCache.set(token, null);
  return null;
}

// ─── MAIN PREPROCESSING PIPELINE ─────────────────────────────────────────────

/**
 * Full NLP preprocessing pipeline:
 * 1. Lowercase + unicode normalize
 * 2. Tokenize
 * 3. Stopword removal
 * 4. OOV resolution: direct map → stem → phonetic → fuzzy
 * 5. Returns concepts (normalized) + stems + rawTokens
 */
export function preprocessQuery(text) {
  const input = String(text || "").toLowerCase().trim();

  const rawTokens = input
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(t => t.length > 1);

  const concepts = new Set();
  const stems = new Set();

  for (const token of rawTokens) {
    if (STOPWORDS.has(token)) continue;

    const concept = resolveToken(token);
    if (concept !== null && concept !== undefined) {
      concepts.add(concept);
    } else if (concept === undefined) {
      // Truly unknown — keep raw + stem as fallback
      const stem = PorterStemmer.stem(token);
      concepts.add(token);
      stems.add(stem);
    }
    // concept === null means explicitly ignored word — skip it
  }

  return { concepts: Array.from(concepts), stems: Array.from(stems), rawTokens };
}

// ─── INTENT DETECTION ────────────────────────────────────────────────────────

const INTENT_PATTERNS = {
  location:    /\b(where|kahan|kidhar|location|address|situated|located|distance|far|reach|direction|how far|kitni door)\b/i,
  fees:        /\b(fee|fees|cost|price|kitna|kitni|kharcha|paisa|tuition|payment|rupee|lakh|total|how much|kitna lagega)\b/i,
  scholarship: /\b(scholarship|discount|waiver|concession|free|financial|merit|aid|get.*scholarship|scholarship.*get|fee.*off|percent.*off|can.*get|get.*fee)\b/i,
  placement:   /\b(placement|job|salary|package|ctc|company|companies|recruiter|hiring|kaunsi|aati|scene|naukri|recruit|google|deloitte)\b/i,
  admission:   /\b(admission|apply|form|process|register|eligibility|documents|join|enroll|how.*apply|apply.*how|without.*jee|jee.*without|take admission|get admission|want.*admission)\b/i,
  courses:     /\b(course|courses|branch|stream|kaunse|which|offer|available|padhai|specialization|what.*course|tell.*about.*cse|all.*program|programs.*available|available.*program)\b/i,
  hostel:      /\b(hostel|accommodation|room|stay|rehna|mess|food|pg|ac|included|facility)\b/i,
  campus:      /\b(campus|library|lab|gym|sports|cafeteria|facilities|amenities|infrastructure|mess|canteen|wifi|equipment|classroom)\b/i,
  contact:     /\b(contact|phone|email|call|reach|number|address|where|far|distance|location)\b/i,
  about:       /\b(about|tell|explain|describe|overview|info|information|batao|what is|kya hai|what.*bmu|bmu.*what|why bmu|why choose|alag|khasiyat)\b/i
};

export function detectIntent(text) {
  const input = String(text || "").toLowerCase();
  const intents = [];
  for (const [intent, pattern] of Object.entries(INTENT_PATTERNS)) {
    if (pattern.test(input)) intents.push(intent);
  }
  return intents.length > 0 ? intents : ["general"];
}

// ─── TOPIC SCORING ───────────────────────────────────────────────────────────

export function scoreTopicNLP(topicKeywords, queryResult) {
  const { concepts, stems } = queryResult;
  const conceptSet = new Set(concepts);
  const stemSet = new Set(stems);

  let score = 0;
  for (const keyword of (topicKeywords || [])) {
    const kw = String(keyword).toLowerCase();
    const kwConcept = CONCEPT_MAP[kw];
    const kwStem = PorterStemmer.stem(kw);

    if (conceptSet.has(kw)) { score += 1; continue; }
    if (kwConcept && conceptSet.has(kwConcept)) { score += 1; continue; }
    if (stemSet.has(kwStem)) { score += 0.8; continue; }
  }
  return score;
}

export function scoreTopicWithIntent(topicKeywords, topicId, queryResult, detectedIntents) {
  let score = scoreTopicNLP(topicKeywords, queryResult);
  const conceptSet = new Set(queryResult.concepts);

  for (const intent of detectedIntents) {
    if (topicId.includes(intent)) score += 2;
  }

  // When score ties, prefer more specific topics over university_overview
  if (topicId === "university_overview") {
    score -= 0.1;
  }
  // When any specific intent is detected, heavily penalize university_overview
  if (topicId === "university_overview" && detectedIntents.some(i =>
    ["fees","scholarship","placement","courses","admission","hostel","contact"].includes(i)
  )) {
    score = Math.max(0, score - 8);
  }
  // Also penalize university_overview when specific concepts are present
  if (topicId === "university_overview" && (
    queryResult.concepts.includes("admission") ||
    queryResult.concepts.includes("fee") ||
    queryResult.concepts.includes("scholarship") ||
    queryResult.concepts.includes("placement") ||
    queryResult.concepts.includes("hostel") ||
    queryResult.concepts.includes("course") ||
    queryResult.concepts.includes("law") ||
    queryResult.concepts.includes("cse") ||
    queryResult.concepts.includes("mba")
  )) {
    score = Math.max(0, score - 8);
  }
  // Penalty: contact_info should not win when intent is clearly something else
  if (topicId === "contact_info" && detectedIntents.some(i =>
    ["hostel","fees","scholarship","placement","courses","admission"].includes(i)
  )) {
    score = Math.max(0, score - 3);
  }

  // Boost hostel when hostel intent is primary
  if (topicId === "hostel" && detectedIntents.includes("hostel")) {
    score += 3;
  }
  // Penalty: total_fees_with_hostel should not win for "is food included" type queries
  if (topicId === "total_fees_with_hostel" && detectedIntents.includes("hostel") &&
      queryResult.rawTokens.some(t => /food|included|meal|khana|ac|room|facility/i.test(t))) {
    score = Math.max(0, score - 5);
  }
  // Penalty: total_fees_with_hostel should not win for pure hostel queries
  if (topicId === "total_fees_with_hostel" && detectedIntents.includes("hostel") && !detectedIntents.includes("fees")) {
    score = Math.max(0, score - 4);
  }
  // Boost campus_facilities when campus intent detected
  if (topicId === "campus_facilities" && (detectedIntents.includes("campus") ||
      queryResult.concepts.includes("campus"))) {
    score += 4;
  }
  // Boost new topics using their dedicated concepts
  if (topicId === "accreditation" && queryResult.concepts.includes("accreditation")) {
    score += 6;
  }
  if (topicId === "why_bmu" && queryResult.concepts.includes("why_bmu")) {
    score += 6;
  }
  if (topicId === "safety_campus" && queryResult.concepts.includes("safety")) {
    score += 6;
  }
  if (topicId === "extracurricular" && queryResult.concepts.includes("extracurricular")) {
    score += 6;
  }
  if (topicId === "internship_industry" && queryResult.concepts.includes("internship")) {
    score += 6;
  }
  // Penalty: placements should not win for internship queries
  if (topicId === "placements" && queryResult.concepts.includes("internship")) {
    score = Math.max(0, score - 8);
  }
  // Boost law_programs when law concept present
  if (topicId === "law_programs" && queryResult.concepts.includes("law")) {
    score += 4;
  }
  // Boost courses_overview when course concept present with about/tell intent
  // BUT only when no specific program is mentioned
  if (topicId === "courses_overview" && queryResult.concepts.includes("course") &&
      !queryResult.concepts.includes("cse") && !queryResult.concepts.includes("mba") &&
      !queryResult.concepts.includes("law") && !queryResult.concepts.includes("btech") &&
      !queryResult.concepts.includes("mechanical") &&
      (detectedIntents.includes("about") || detectedIntents.includes("courses") ||
       queryResult.rawTokens.some(t => /tell|about|all|list|what|which/i.test(t)))) {
    score += 5;
  }
  // Boost courses_overview for "all programs/courses" queries
  if (topicId === "courses_overview" &&
      queryResult.rawTokens.some(t => /\ball\b/i.test(t)) &&
      !queryResult.concepts.includes("cse") && !queryResult.concepts.includes("mba") &&
      !queryResult.concepts.includes("law") && !queryResult.concepts.includes("btech")) {
    score += 5;
  }
  // Boost mba_program strongly when mba concept present with fee intent
  if (topicId === "mba_program" && queryResult.concepts.includes("mba")) {
    score += 5;
  }
  // Penalty: btech_fees should not win for MBA fee queries
  if (topicId === "btech_fees" && queryResult.concepts.includes("mba")) {
    score = Math.max(0, score - 6);
  }
  // Penalty: total_fees_with_hostel should not win for "is food included" type queries
  if (topicId === "total_fees_with_hostel" && detectedIntents.includes("hostel") &&
      queryResult.rawTokens.some(t => /food|included|meal|khana|ac|room|facility/i.test(t))) {
    score = Math.max(0, score - 5);
  }
  // Boost btech_cse when cse concept present with about/details intent
  if (topicId === "btech_cse" && queryResult.concepts.includes("cse") &&
      (detectedIntents.includes("about") || detectedIntents.includes("courses") ||
       queryResult.rawTokens.some(t => /tell|about|describe|explain|info|detail/i.test(t)))) {
    score += 5;
  }
  // Penalty: btech_cse should not win for fee queries
  if (topicId === "btech_cse" && detectedIntents.includes("fees") && !detectedIntents.includes("courses")) {
    score = Math.max(0, score - 4);
  }
  // Penalty: placements should not win when query is about a specific course/program
  if (topicId === "placements" && queryResult.concepts.includes("cse") &&
      queryResult.rawTokens.some(t => /tell|about|program|course|branch|detail/i.test(t)) &&
      !queryResult.rawTokens.some(t => /placement|job|salary|package|recruit|company/i.test(t))) {
    score = Math.max(0, score - 10);
  }
  // Boost btech_overview for branch/overview queries
  if (topicId === "btech_overview" && queryResult.concepts.includes("btech") &&
      queryResult.rawTokens.some(t => /branch|branches|available|overview|all/i.test(t))) {
    score += 4;
  }
  // Penalty: courses_overview should not win when specific program mentioned
  if (topicId === "courses_overview" && (
    queryResult.concepts.includes("cse") || queryResult.concepts.includes("mba") ||
    queryResult.concepts.includes("law") || queryResult.concepts.includes("mechanical") ||
    queryResult.concepts.includes("scholarship") || queryResult.concepts.includes("placement") ||
    queryResult.concepts.includes("hostel") || queryResult.concepts.includes("fee")
  )) {
    score = Math.max(0, score - 4);
  }
  // Boost total_fees_with_hostel for "total/cost/how much/with hostel" queries
  if (topicId === "total_fees_with_hostel" && queryResult.rawTokens.some(t =>
    /total|overall|annual|yearly|complete|4.year|four.year|with.hostel/i.test(t))) {
    score += 4;
  }
  // Penalty: btech_fees should not win when "total" is in query
  if (topicId === "btech_fees" && queryResult.rawTokens.some(t => /\btotal\b/i.test(t))) {
    score = Math.max(0, score - 5);
  }
  // Boost btech_fees when btech/cse + fee present but NOT total
  if (topicId === "btech_fees" && (queryResult.concepts.includes("btech") || queryResult.concepts.includes("cse")) &&
      queryResult.concepts.includes("fee") &&
      !queryResult.rawTokens.some(t => /\btotal\b/i.test(t))) {
    score += 3;
  }
  // Boost documents_required strongly when documents concept present
  if (topicId === "documents_required" && queryResult.concepts.includes("documents")) {
    score += 6;
  }
  return score;
}

// ─── HYBRID INTENT DETECTION (ML + Regex Fallback) ───────────────────────────
//
// This function is the NEW entry point for intent detection.
// It tries the ML model first (via Python service).
// If ML is unavailable or returns low confidence, it falls back
// to the original regex-based detectIntent() — keeping the system stable.
//
// The original detectIntent() function above is UNTOUCHED and still works
// independently as a reliable fallback.

export async function detectIntentHybrid(text) {
  try {
    // Lazy import to avoid circular deps — only loaded when called
    const { predictIntentML } = await import("./pythonNlpService.js");
    const mlResult = await predictIntentML(text);

    if (mlResult && mlResult.intent && mlResult.confidence >= 0.45) {
      // ML model is confident — use its prediction
      // Map single ML intent to array format expected by the rest of the pipeline
      const mlIntent = mlResult.intent;
      const regexIntents = detectIntent(text);

      // Merge: ML primary + any extra regex intents (multi-intent support)
      const merged = [mlIntent, ...regexIntents.filter(i => i !== mlIntent && i !== "general")];
      console.log(`[HybridIntent] ML="${mlIntent}"(${mlResult.confidence}) | Regex=[${regexIntents}] | Final=[${merged}]`);
      return merged;
    }
  } catch (err) {
    // Python service offline or error — silently fallback
    console.log(`[HybridIntent] ML unavailable (${err.message}) — using regex fallback`);
  }

  // Fallback: use original regex-based detection
  return detectIntent(text);
}
