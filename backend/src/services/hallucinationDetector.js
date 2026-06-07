/**
 * Hallucination Detector
 * Checks if LLM-generated answer contradicts known BMU facts
 * Flags suspicious numbers/claims that don't match our knowledge base
 */

// Known ground truth facts — numbers the LLM must not contradict
const GROUND_TRUTH = {
  fees: {
    btech_cse_per_year: 410000,
    btech_me_ecome_per_year: 365000,
    mba_per_year: 755000,
    ba_llb_per_year: 390000,
    llb_per_year: 250000,
    bba_per_year: 350000,
    hostel_double_per_year: 236000,
    hostel_triple_per_year: 225000
  },
  placements: {
    highest_ctc_lpa: 59.35,
    top10_avg_lpa: 19.38,
    top25_avg_lpa: 14.09
  },
  scholarships: {
    above_90_percent: 50,   // 50% waiver
    above_85_percent: 30,   // 30% waiver
    above_80_percent: 20    // 20% waiver
  },
  location: {
    distance_from_delhi_km: 45
  }
};

// Patterns to extract numbers from LLM response
const NUMBER_PATTERNS = [
  // Fee patterns
  { regex: /(\d+(?:\.\d+)?)\s*lakh[s]?\s*per\s*year.*cse/i, fact: "btech_cse_per_year", multiplier: 100000 },
  { regex: /cse.*(\d+(?:\.\d+)?)\s*lakh[s]?\s*per\s*year/i, fact: "btech_cse_per_year", multiplier: 100000 },
  // Placement patterns
  { regex: /highest.*?(\d+(?:\.\d+)?)\s*lpa/i, fact: "highest_ctc_lpa", multiplier: 1 },
  { regex: /(\d+(?:\.\d+)?)\s*lpa.*highest/i, fact: "highest_ctc_lpa", multiplier: 1 },
  // Scholarship patterns
  { regex: /90\s*(?:percent|%)[^.]*?(\d+)\s*(?:percent|%)\s*(?:off|waiver)/i, fact: "above_90_percent", multiplier: 1 },
  // Distance patterns
  { regex: /(\d+)\s*(?:km|kilometre|kilometer).*delhi/i, fact: "distance_from_delhi_km", multiplier: 1 },
  { regex: /delhi.*(\d+)\s*(?:km|kilometre|kilometer)/i, fact: "distance_from_delhi_km", multiplier: 1 }
];

// Flat lookup for ground truth values
const FLAT_TRUTH = {
  btech_cse_per_year: GROUND_TRUTH.fees.btech_cse_per_year,
  btech_me_ecome_per_year: GROUND_TRUTH.fees.btech_me_ecome_per_year,
  mba_per_year: GROUND_TRUTH.fees.mba_per_year,
  highest_ctc_lpa: GROUND_TRUTH.placements.highest_ctc_lpa,
  top10_avg_lpa: GROUND_TRUTH.placements.top10_avg_lpa,
  above_90_percent: GROUND_TRUTH.scholarships.above_90_percent,
  above_85_percent: GROUND_TRUTH.scholarships.above_85_percent,
  distance_from_delhi_km: GROUND_TRUTH.location.distance_from_delhi_km
};

/**
 * Check LLM response for hallucinated numbers
 * Returns { isHallucination: bool, issues: string[] }
 */
export function detectHallucination(llmResponse) {
  const issues = [];
  const text = String(llmResponse || "");

  for (const pattern of NUMBER_PATTERNS) {
    const match = text.match(pattern.regex);
    if (!match) continue;

    const extractedValue = parseFloat(match[1]) * pattern.multiplier;
    const truthValue = FLAT_TRUTH[pattern.fact];
    if (!truthValue) continue;

    // Allow 10% tolerance for rounding
    const tolerance = truthValue * 0.10;
    if (Math.abs(extractedValue - truthValue) > tolerance) {
      issues.push(`Possible wrong ${pattern.fact}: said ${extractedValue}, truth is ${truthValue}`);
    }
  }

  // Check for obviously wrong claims
  if (/100\s*%\s*placement/i.test(text)) {
    issues.push("Claims 100% placement rate — not verified");
  }
  if (/iit|nit\b/i.test(text) && /bmu|bml/i.test(text)) {
    issues.push("Comparing BMU to IIT/NIT — potentially misleading");
  }

  return {
    isHallucination: issues.length > 0,
    issues,
    confidence: issues.length === 0 ? 1.0 : Math.max(0.3, 1.0 - issues.length * 0.25)
  };
}
