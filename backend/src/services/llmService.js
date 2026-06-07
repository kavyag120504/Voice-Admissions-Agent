import { config } from "../config.js";

const systemPrompt = `
You are Aria, an admissions counselor at BML Munjal University. You are talking to students and parents on a phone call.

MOST IMPORTANT RULE: You are a HUMAN. Never say you are an AI, assistant, bot, or language model. If anyone asks if you are an AI or robot, say something like "No no, I'm Aria, I work in the admissions team here at BMU." and move on naturally.

HOW YOU TALK — like a real person on a phone call:
- Use natural fillers: "So", "Actually", "You know", "Honestly", "Look", "See", "Right so"
- Hindi fillers: "Dekho", "Suno", "Actually", "Basically", "Toh", "Haan so"
- Start sentences naturally: "So the thing is...", "Honestly...", "Look, basically..."
- React like a human: "Oh great!", "That's a good score actually", "Yeah absolutely"
- Ask follow-up questions like you genuinely care: "By the way, which city are you from?", "Have you given JEE?"
- Sometimes be slightly informal: "it's pretty good", "honestly it's worth it", "you'll love the campus"
- Never use bullet points or numbered lists — just talk naturally
- Keep it SHORT — 2 to 3 sentences max, like a real phone conversation
- Never sound like you're reading from a brochure

LANGUAGE:
- If they speak Hindi or Hinglish → reply in natural Hinglish the way Indians actually talk on phone
- If they speak English → reply in simple conversational Indian English
- Mix naturally: "Haan so the fees are around 4 lakh per year, which is quite reasonable actually"

RESPOND BASED ON USER EMOTION (UserEmotion field tells you this):
- stressed/worried → Be extra calm and reassuring. Start with "Don't worry," or "Bilkul tension mat lo,". Slow down, be simple.
- sad/upset → Be warm and empathetic first. "I understand, let me help you with this." Don't rush.
- angry/frustrated → Stay calm, don't match their energy. Acknowledge first: "I hear you, let me sort this out."
- happy/excited → Match their energy! Be enthusiastic. "Oh that's great!" or "Haan bilkul!"
- neutral → Normal friendly tone.

BMU FACTS (use these, never make up numbers):
- Location: Gurugram, Haryana, about 45 km from Delhi on NH-48
- Hero Group backed, NAAC A grade, QS Diamond Rating, ranked 5th private engineering university
- B.Tech branches: CSE (4.1L/yr), Mechanical and EComE (3.65L/yr) — 4 years, need 60% in PCM
- MBA: 7.55L/yr, 2 years, CAT/MAT/XAT accepted
- BA LLB / BBA LLB: 3.9L/yr, 5 years, 45% in 12th, CLAT accepted
- LLB: 2.5L/yr, 3 years
- BBA: 3.5L/yr, B.Com: 2.5L/yr
- Hostel: 2.25 to 2.36L/yr, AC rooms, food included
- Scholarships: 90%+ gets 50% off fees, 85-90% gets 30% off, 80-85% gets 20% off
- Top entrance rankers get 100% fee waiver
- Girls get extra 10% scholarship
- Placements: highest 59.35 LPA, top 10% average 19.38 LPA
- Recruiters: Google, Deloitte, HCL, L&T, KPMG, Tata Digital, MG Motors
- Entrance: JEE/CUET/BMU-SAT for BTech, CLAT for Law, CAT/MAT for MBA
- Contact: +91-1275-286101, admissions@bmu.edu.in

WHEN YOU DON'T KNOW SOMETHING:
Don't say "I don't have that information". Instead say something like:
"You know what, let me connect you with our team for that specific detail — they'll give you the exact answer. You can call us at plus 91 1275 286101. Anything else I can help with?"

WHEN ASKED NON-BMU QUESTIONS:
Gently redirect: "Haha I can only really speak about BMU to be honest — that's where I work! Is there something about our programs I can help with?"

EXAMPLE RESPONSES:

User: "What courses do you have?"
Aria: "So we have Engineering, Management, Law, and Liberal Arts mainly. CSE is honestly the most popular — it has AI and Data Science specializations. Which field are you thinking about?"

User: "BTech CSE fees kitni hai?"
Aria: "Haan so CSE ki fees around 4.1 lakh per year hai. Aur agar hostel bhi chahiye toh woh alag se roughly 2.3 lakh hoga, usme khana bhi included hai. Aapke 12th mein kitne percent hain? Scholarship bhi mil sakti hai."

User: "Scholarship milegi kya?"
Aria: "Oh absolutely! Agar 90 percent ya usse zyada hain toh 50 percent fees off milti hai — that's huge honestly. 85 se 90 ke beech hai toh 30 percent off. Aapke kitne marks hain?"

User: "Are you an AI?"
Aria: "No no, I'm Aria, I'm part of the admissions team here at BMU. Is there something about our programs I can help you with?"
`;


function buildPrompt({
  turns,
  userText,
  emotion,
  languageMode,
  conversationSummary = "",
  retrievalSnippets = [],
  dialogDirective = "",
  entityContext = "general"
}) {
  const emotionHint = `UserEmotion=${emotion.emotion}, Confidence=${emotion.confidence}`;
  const languageHint = `LanguageMode=${languageMode}. If hindi or hinglish, reply in natural Hinglish.`;

  const history = turns
    .slice(-config.maxContextTurns)
    .map((turn) => `${turn.role.toUpperCase()}: ${turn.content}`)
    .join("\n");

  const retrievalContext = retrievalSnippets.length
    ? retrievalSnippets.map((s, idx) => `[R${idx + 1}] ${s.text}\nSource=${s.source}`).join("\n\n")
    : "none";

  return `${systemPrompt}\n${emotionHint}\n${languageHint}\n\nExtractedEntities: ${entityContext}\nDialogDirective: ${dialogDirective || "none"}\nConversationSummary: ${conversationSummary || "none"}\n\nKnowledgeContext:\n${retrievalContext}\n\nConversationHistory:\n${history}\n\nUSER: ${userText}\nASSISTANT:`;
}

let resolvedModelCache = {
  model: null,
  checkedAt: 0
};

async function withTimeoutFetch(url, options = {}, timeoutMs = 14000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function resolveModel() {
  const now = Date.now();
  if (resolvedModelCache.model && now - resolvedModelCache.checkedAt < 60_000) {
    return resolvedModelCache.model;
  }

  let response;
  try {
    response = await withTimeoutFetch(`${config.ollamaBaseUrl}/api/tags`, { method: "GET" }, 5000);
  } catch {
    throw new Error("ollama_unreachable");
  }

  if (!response.ok) throw new Error(`ollama_tags_failed_${response.status}`);
  const data = await response.json();
  const models = Array.isArray(data.models) ? data.models : [];
  const names = models.map((m) => m.name).filter(Boolean);

  if (!names.length) throw new Error("ollama_no_models");
  let selected = names.includes(config.ollamaModel) ? config.ollamaModel : null;
  if (!selected && config.preferredModels.length) {
    for (const preferred of config.preferredModels) {
      const exact = names.find((n) => n.toLowerCase() === preferred.toLowerCase());
      if (exact) {
        selected = exact;
        break;
      }
      const partial = names.find((n) => n.toLowerCase().includes(preferred.toLowerCase()));
      if (partial) {
        selected = partial;
        break;
      }
    }
  }
  if (!selected) selected = names[0];
  resolvedModelCache = { model: selected, checkedAt: now };
  return selected;
}

async function streamFromOllama(prompt, onChunk, abortSignal, overrideModel) {
  const model = overrideModel || (await resolveModel());
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.requestTimeoutMs);
  const onAbort = () => controller.abort();
  abortSignal?.addEventListener("abort", onAbort);

  let response;
  try {
    response = await fetch(`${config.ollamaBaseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: true,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          num_ctx: 2048,
          num_predict: 120,
          repeat_penalty: 1.1
        }
      }),
      signal: controller.signal
    });
  } catch (error) {
    if (abortSignal?.aborted) throw new Error("generation_aborted");
    if (String(error).includes("AbortError")) throw new Error("ollama_timeout");
    throw new Error("ollama_unreachable");
  } finally {
    clearTimeout(timeoutId);
    abortSignal?.removeEventListener("abort", onAbort);
  }

  if (!response.ok || !response.body) {
    throw new Error(`ollama_generate_failed_${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let done = false;
  let buffer = "";
  let fullText = "";

  while (!done) {
    if (abortSignal?.aborted) throw new Error("generation_aborted");
    const chunk = await reader.read();
    done = chunk.done;
    if (chunk.value) {
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        let json;
        try {
          json = JSON.parse(line);
        } catch {
          continue;
        }
        if (json.error) throw new Error(`ollama_runtime_error_${json.error}`);
        const token = json.response || "";
        fullText += token;
        if (token) onChunk(token);
      }
    }
  }

  return fullText.trim();
}

export const llmService = {
  createAbortController() {
    return new AbortController();
  },
  async streamReply(payload, onChunk, abortSignal) {
    const prompt = buildPrompt(payload);
    try {
      return await streamFromOllama(prompt, onChunk, abortSignal);
    } catch (error) {
      const text = String(error || "");
      // If primary model times out, retry once on a likely faster fallback.
      if (text.includes("ollama_timeout") && config.preferredModels.length > 1) {
        return streamFromOllama(prompt, onChunk, abortSignal, config.preferredModels[1]);
      }
      throw error;
    }
  },
  postProcessReply({ userText, replyText, languageMode, retrievalConfidence = 0, dialogDirective = "" }) {
    const user = String(userText || "").trim();
    let reply = String(replyText || "").replace(/\s+/g, " ").trim();
    if (!reply) return reply;

    reply = reply
      .replace(/\b(i('| a)m|my name is)\s+aria\b/gi, "")
      .replace(/\badmission counselor\b/gi, "")
      .replace(/\s{2,}/g, " ")
      .trim();

    const norm = (t) =>
      String(t || "")
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
    const words = (t) => norm(t).split(" ").filter((w) => w.length > 2);
    const uw = words(user);
    const rw = words(reply);
    const setU = new Set(uw);
    const setR = new Set(rw);
    let inter = 0;
    for (const w of setU) if (setR.has(w)) inter += 1;
    const overlap = setU.size ? inter / setU.size : 0;

    const hindiMode = languageMode === "hindi" || languageMode === "hinglish";
    const hasQuestion = /[?？]/.test(reply);
    const followup = hindiMode
      ? "क्या आप course details, fees, scholarship, या admission steps में से कौन सा पहले जानना चाहेंगे?"
      : "Would you like details on courses, fees, scholarships, or admission steps first?";

    if (overlap > 0.72) {
      return hindiMode
        ? `समझ गया। मैं आपकी बात दोहराने के बजाय सीधे मदद करता हूँ। ${followup}`
        : `Got it. Instead of repeating your words, I will help directly. ${followup}`;
    }

    if (retrievalConfidence < 0.25) {
      return hindiMode
        ? `मुझे सटीक जानकारी देने के लिए एक छोटी clarification चाहिए। ${followup}`
        : `To give accurate information, I need one quick clarification. ${followup}`;
    }

    if (/more information|more details|aur jankari|और जानकारी/i.test(user)) {
      return hindiMode
        ? "ज़रूर। मैं detail में बताता हूँ। पहले बताइए आप course, fees, scholarship, या admission timeline में से किस पर detail चाहते हैं।"
        : "Sure. I can explain in detail. Please tell me whether you want deep details on courses, fees, scholarships, or admission timeline first.";
    }

    const transitionHint = /topic_shift|combined_request/i.test(dialogDirective || "");
    if (transitionHint && !/^got it|^समझ गया|^ठीक है/i.test(reply)) {
      reply = hindiMode ? `ठीक है, topic बदल गया है। ${reply}` : `Got it, we are shifting topics. ${reply}`;
    }

    if (!hasQuestion) reply = `${reply} ${followup}`;
    return reply;
  }
};
