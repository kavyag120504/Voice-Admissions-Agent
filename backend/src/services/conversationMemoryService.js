/**
 * Conversation Memory Service
 * Builds rolling summaries and resolves contextual references
 *
 * Context-aware dialogue example:
 *   Turn 1: "What is CSE fees?" → context: {program: "cse"}
 *   Turn 2: "What about hostel?" → resolved: "What about hostel for CSE?"
 *   Turn 3: "And scholarship?" → resolved: "And scholarship for CSE?"
 */

function clip(text, max = 180) {
  const s = String(text || "").replace(/\s+/g, " ").trim();
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}...`;
}

// Extract the active program/topic from recent turns
function extractActiveContext(turns) {
  const recent = turns.slice(-6);
  const context = {
    program: null,
    topic: null,
    lastQuestion: null
  };

  for (const turn of recent.reverse()) {
    const text = String(turn.content || "").toLowerCase();

    // Detect program mentions
    if (!context.program) {
      if (/\bcse\b|computer science/.test(text)) context.program = "CSE";
      else if (/\bmba\b/.test(text)) context.program = "MBA";
      else if (/\bmechanical\b|\bmech\b/.test(text)) context.program = "Mechanical";
      else if (/\bllb\b|law/.test(text)) context.program = "Law";
      else if (/\bbba\b/.test(text)) context.program = "BBA";
      else if (/\become\b|electronics/.test(text)) context.program = "EComE";
    }

    // Detect topic mentions
    if (!context.topic) {
      if (/fee|fees|cost|paisa/.test(text)) context.topic = "fees";
      else if (/hostel|accommodation/.test(text)) context.topic = "hostel";
      else if (/scholarship|discount/.test(text)) context.topic = "scholarship";
      else if (/placement|job/.test(text)) context.topic = "placement";
      else if (/admission|apply/.test(text)) context.topic = "admission";
    }
  }

  return context;
}

/**
 * Resolve contextual references in a query
 * "What about hostel?" → "What about hostel for CSE?" (if CSE was mentioned before)
 * "And scholarship?" → "And scholarship for CSE?" (carries program context)
 */
export function resolveContextualQuery(userText, turns) {
  const text = String(userText || "").trim();
  const ctx = extractActiveContext(turns);

  // Short follow-up queries that need context injection
  const isFollowUp = /^(what about|and|also|how about|tell me about|aur|aur kya|uske baare mein)\b/i.test(text)
    || text.split(" ").length <= 4;

  if (!isFollowUp || !ctx.program) return text;

  // Don't inject if program already mentioned in query
  const programMentioned = /cse|mba|mechanical|law|bba|ecome/i.test(text);
  if (programMentioned) return text;

  // Inject program context
  return `${text} for ${ctx.program}`;
}

function summarizeTurns(turns) {
  const recent = turns.slice(-16);
  const userTurns = recent.filter(t => t.role === "user").map(t => t.content);
  const assistantTurns = recent.filter(t => t.role === "assistant").map(t => t.content);
  const ctx = extractActiveContext(recent);

  const latestUserIntent = userTurns.length ? clip(userTurns[userTurns.length - 1], 120) : "";
  const recurringTopics = [];
  const corpus = userTurns.join(" ").toLowerCase();

  if (/(course|program|branch|कोर्स|ब्रांच|admission|एडमिशन)/.test(corpus)) recurringTopics.push("admission_or_courses");
  if (/(fee|fees|payment|फीस)/.test(corpus)) recurringTopics.push("fees");
  if (/(hostel|transport|placement|exam|प्लेसमेंट|एग्जाम)/.test(corpus)) recurringTopics.push("campus_life");

  const contextHint = ctx.program
    ? `ActiveProgram: ${ctx.program}${ctx.topic ? `, ActiveTopic: ${ctx.topic}` : ""}`
    : "ActiveProgram: not_set";

  return [
    `RecentUserIntent: ${latestUserIntent || "not_set"}`,
    `RecurringTopics: ${recurringTopics.length ? recurringTopics.join(", ") : "none"}`,
    `RecentAssistantReply: ${assistantTurns.length ? clip(assistantTurns[assistantTurns.length - 1], 120) : "not_set"}`,
    contextHint
  ].join("\n");
}

export const conversationMemoryService = {
  buildSummary(turns) {
    return summarizeTurns(turns);
  },
  resolveContextualQuery(userText, turns) {
    return resolveContextualQuery(userText, turns);
  }
};
