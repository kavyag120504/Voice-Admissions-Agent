/**
 * Entity Extraction — pulls structured entities from raw query text
 * Extracts: program, specialization, year, percentage, hostel preference, language
 */

const PROGRAM_PATTERNS = [
  { pattern: /\bb\.?tech|btech|engineering\b/i, entity: "btech" },
  { pattern: /\bcse|computer science|computer engineering\b/i, entity: "cse" },
  { pattern: /\become|electronics.*computer|ece\b/i, entity: "ecome" },
  { pattern: /\bmechanical|mech\b/i, entity: "mechanical" },
  { pattern: /\bmba|master.*business|business admin\b/i, entity: "mba" },
  { pattern: /\bbba\b/i, entity: "bba" },
  { pattern: /\bb\.?com|bcom|commerce\b/i, entity: "bcom" },
  { pattern: /\bba.*llb|llb.*ba|ba llb\b/i, entity: "ballb" },
  { pattern: /\bbba.*llb|llb.*bba\b/i, entity: "bbllb" },
  { pattern: /\bllb\b/i, entity: "llb" },
  { pattern: /\bm\.?tech|mtech\b/i, entity: "mtech" },
  { pattern: /\bphd|doctorate|research\b/i, entity: "phd" },
  { pattern: /\bliberal arts|ba.*hons\b/i, entity: "liberalarts" },
];

const SPECIALIZATION_PATTERNS = [
  { pattern: /\bai|artificial intelligence\b/i, entity: "ai" },
  { pattern: /\bdata science|data analytics\b/i, entity: "datascience" },
  { pattern: /\bcyber security|cybersecurity\b/i, entity: "cybersecurity" },
  { pattern: /\biot|internet of things\b/i, entity: "iot" },
  { pattern: /\bvlsi\b/i, entity: "vlsi" },
  { pattern: /\brobotic|automation\b/i, entity: "robotics" },
  { pattern: /\bcloud\b/i, entity: "cloud" },
  { pattern: /\bblockchain\b/i, entity: "blockchain" },
];

const YEAR_PATTERN = /\b(1st|2nd|3rd|4th|first|second|third|fourth|1|2|3|4)\s*(year|yr)\b/i;
const PERCENTAGE_PATTERN = /\b(\d{2,3})\s*(%|percent|percentage|marks|score)\b/i;
const HOSTEL_PATTERN = /\bhostel|accommodation|stay|room|rehna\b/i;
const SCHOLARSHIP_PATTERN = /\bscholarship|discount|waiver|concession|free\b/i;

/**
 * Extract all entities from query text
 */
export function extractEntities(text) {
  const input = String(text || "");
  const entities = {
    programs: [],
    specializations: [],
    year: null,
    percentage: null,
    needsHostel: false,
    needsScholarship: false
  };

  // Extract programs
  for (const { pattern, entity } of PROGRAM_PATTERNS) {
    if (pattern.test(input)) entities.programs.push(entity);
  }

  // Extract specializations
  for (const { pattern, entity } of SPECIALIZATION_PATTERNS) {
    if (pattern.test(input)) entities.specializations.push(entity);
  }

  // Extract year
  const yearMatch = input.match(YEAR_PATTERN);
  if (yearMatch) entities.year = yearMatch[1];

  // Extract percentage
  const pctMatch = input.match(PERCENTAGE_PATTERN);
  if (pctMatch) entities.percentage = Number(pctMatch[1]);

  // Flags
  entities.needsHostel = HOSTEL_PATTERN.test(input);
  entities.needsScholarship = SCHOLARSHIP_PATTERN.test(input);

  return entities;
}

/**
 * Generate a contextual response hint based on extracted entities
 * This helps the LLM give more targeted answers
 */
export function buildEntityContext(entities) {
  const parts = [];
  if (entities.programs.length) parts.push(`Program: ${entities.programs.join(", ")}`);
  if (entities.specializations.length) parts.push(`Specialization: ${entities.specializations.join(", ")}`);
  if (entities.year) parts.push(`Year: ${entities.year}`);
  if (entities.percentage !== null) parts.push(`Student percentage: ${entities.percentage}%`);
  if (entities.needsHostel) parts.push("Needs hostel info");
  if (entities.needsScholarship) parts.push("Needs scholarship info");
  return parts.length ? parts.join(" | ") : "general";
}
