import { emotionService } from "../services/emotionService.js";
import { llmService } from "../services/llmService.js";
import { responsePolicy } from "../services/responsePolicy.js";
import { conversationMemoryService } from "../services/conversationMemoryService.js";
import { tfidfRetrieve } from "../services/tfidfService.js";
import { responseQualityService } from "../services/responseQualityService.js";
import { runtimeMetricsService } from "../services/runtimeMetricsService.js";
import { sessionManager } from "./sessionManager.js";
import { spellCorrect } from "../services/spellCorrectionService.js";
import { extractEntities, buildEntityContext } from "../services/entityExtractionService.js";
import { findSimilarTopic } from "../services/semanticSimilarityService.js";
import { detectHallucination } from "../services/hallucinationDetector.js";
import { pythonNlpService } from "../services/pythonNlpService.js";
import { config } from "../config.js";

function send(ws, payload) {
  ws.send(JSON.stringify(payload));
}

function buildRealtimeAck(isHindiMode, cleanText) {
  const asksFees = /fee|fees|tuition|फीस|payment/i.test(cleanText);
  const asksScholarship = /scholarship|स्कॉलरशिप|aid/i.test(cleanText);
  const asksCourses = /course|courses|program|कोर्स|प्रोग्राम|btech|cse|ece|mechanical|civil/i.test(cleanText);

  if (isHindiMode) {
    if (asksFees && asksScholarship) return "ठीक है, मैं fees और scholarship दोनों चेक कर रहा हूँ।";
    if (asksFees) return "ठीक है, मैं fees details निकाल रहा हूँ।";
    if (asksCourses) return "ठीक है, मैं courses details बता रहा हूँ।";
    return "ठीक है, एक सेकंड, मैं सटीक जवाब दे रहा हूँ।";
  }

  if (asksFees && asksScholarship) return "Got it. Checking both fees and scholarship details now.";
  if (asksFees) return "Got it. Pulling fee details now.";
  if (asksCourses) return "Got it. Pulling course details now.";
  return "Got it. One moment, I am preparing an accurate answer.";
}

function mapErrorToHint(errorText) {
  if (errorText.includes("ollama_unreachable")) {
    return "Ollama is not reachable. Start Ollama app/service locally.";
  }
  if (errorText.includes("ollama_no_models")) {
    return "No local Ollama model found. Run: ollama pull llama3.1:8b-instruct-q4_K_M";
  }
  if (errorText.includes("ollama_timeout")) {
    return "Model response timed out. Use a smaller model or retry.";
  }
  if (errorText.includes("runtime_error")) {
    return "Ollama runtime model error. Check model name and pull it again.";
  }
  return "LLM engine error. Check Ollama status and model availability.";
}

function inferTopics(text) {
  const normalized = String(text || "").toLowerCase();
  const topics = new Set();
  if (/(course|courses|program|branch|कोर्स|ब्रांच|प्रोग्राम|btech|b\.tech|cse|ece|mechanical|civil)/i.test(normalized)) {
    topics.add("courses");
  }
  if (/(fee|fees|tuition|फीस|payment|scholarship)/i.test(normalized)) {
    topics.add("fees");
  }
  if (/(scholarship|scholarships|स्कॉलरशिप|financial aid|aid)/i.test(normalized)) {
    topics.add("scholarships");
  }
  if (/(admission|apply|eligibility|एडमिशन|अप्लाई)/i.test(normalized)) {
    topics.add("admission");
  }
  if (/(hostel|transport|campus|होस्टल|कैम्पस)/i.test(normalized)) {
    topics.add("campus");
  }
  return Array.from(topics);
}

function isFollowupJoinIntent(text) {
  return /(along with|alongwith|with that|uske sath|उसके साथ|साथ में|also|plus)/i.test(String(text || "").toLowerCase());
}

function buildDialogDirective(session, userText) {
  const newTopics = inferTopics(userText);
  const previousTopics = session.dialogState?.currentTopics || [];
  const joinIntent = isFollowupJoinIntent(userText);

  let mergedTopics = newTopics;
  let mode = "normal";
  if (joinIntent && previousTopics.length && newTopics.length) {
    mergedTopics = Array.from(new Set([...previousTopics, ...newTopics]));
    mode = "combined_request";
  } else if (!joinIntent && previousTopics.length && newTopics.length) {
    const hasShift = newTopics.some((t) => !previousTopics.includes(t));
    if (hasShift) mode = "topic_shift";
  } else if (joinIntent && previousTopics.length && !newTopics.length) {
    mergedTopics = previousTopics;
    mode = "followup_reference";
  }

  const directive = [
    `mode=${mode}`,
    `currentTopics=${mergedTopics.join(",") || "unknown"}`,
    `previousTopics=${previousTopics.join(",") || "none"}`,
    "Answer with practical next steps, not generic repetition."
  ].join(" | ");

  return { directive, mergedTopics, mode };
}

function getLastAssistantTurn(turns = []) {
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    if (turns[i].role === "assistant") return turns[i].content || "";
  }
  return "";
}

function normalizeWords(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

function isLikelyEchoUtterance(userText, assistantText) {
  const u = normalizeWords(userText);
  const a = new Set(normalizeWords(assistantText));
  if (u.length < 3 || a.size < 6) return false;
  let overlap = 0;
  for (const w of u) if (a.has(w)) overlap += 1;
  return overlap / u.length >= 0.65;
}

function buildDeterministicAnswer(_cleanText, _isHindiMode, _topics, _dialogMode) {
  // Knowledge base (bmu_facts.json) and policy grounding now handle all answers.
  return null;
}

export async function handleRealtimeMessage(ws, incoming) {
  const { type, sessionId, text } = incoming;

  if (type === "session.start") {
    const session = sessionManager.createSession();
    runtimeMetricsService.inc("sessionsStarted");
    send(ws, { type: "session.started", sessionId: session.sessionId });
    return;
  }

  if (type === "session.interrupt") {
    const activeSession = sessionManager.getSession(sessionId);
    if (activeSession && sessionManager.cancelGeneration(sessionId)) {
      runtimeMetricsService.inc("interruptions");
      send(ws, { type: "assistant.interrupted", message: "Interrupted. Listening now." });
    }
    return;
  }

  if (type !== "user.utterance") {
    send(ws, { type: "error", message: "Unknown message type." });
    return;
  }

  const session = sessionManager.getSession(sessionId);
  if (!session) {
    send(ws, { type: "error", message: "Invalid session. Start a new session." });
    return;
  }

  if (session.inFlight) {
    sessionManager.cancelGeneration(sessionId);
    runtimeMetricsService.inc("interruptions");
    send(ws, { type: "assistant.interrupted", message: "Interrupted by new user speech." });
  }

  const cleanText = String(text || "").trim();
  if (!cleanText) {
    send(ws, { type: "assistant.done", text: "Please say your question once again." });
    return;
  }
  if (cleanText.length > config.maxInputChars) {
    send(ws, { type: "assistant.done", text: "Your message is too long for live mode. Please say it in a shorter line." });
    return;
  }
  if (cleanText.length < 3) {
    const msg = "Please say a slightly longer question so I can answer accurately.";
    send(ws, { type: "assistant.done", text: msg });
    return;
  }

  // Step 1: Spell correction
  const spellResult = spellCorrect(cleanText);
  const correctedText = spellResult.text;

  // Step 2: Context-aware query resolution
  // "What about hostel?" after "CSE fees?" → "What about hostel for CSE?"
  const contextResolvedText = conversationMemoryService.resolveContextualQuery(correctedText, session.turns);

  // Step 3: Entity extraction (rule-based)
  const entities = extractEntities(contextResolvedText);
  const entityContext = buildEntityContext(entities);

  // Step 3b: Enhanced NER via Python spaCy (if available)
  pythonNlpService.extractEntitiesNLP(contextResolvedText).then(nerResult => {
    if (nerResult?.bmu_entities) {
      // Merge Python NER results with rule-based entities (non-blocking)
      const bmuEnt = nerResult.bmu_entities;
      if (bmuEnt.percentage && !entities.percentage) entities.percentage = bmuEnt.percentage;
      if (bmuEnt.programs?.length && !entities.programs?.length) entities.programs = bmuEnt.programs;
    }
  }).catch(() => {});

  const emotion = emotionService.analyzeUserSignal(text || "");
  const isHindiMode = emotion.languageMode === "hindi" || emotion.languageMode === "hinglish";
  const lastAssistantTurn = getLastAssistantTurn(session.turns);
  if (isLikelyEchoUtterance(cleanText, lastAssistantTurn)) {
    // Only block if very high overlap AND very recent (within 3 seconds)
    const timeSinceLastTurn = Date.now() - (session.updatedAt || 0);
    if (timeSinceLastTurn < 3000) {
      const prompt = isHindiMode
        ? "Lagta hai audio mein kuch repeat hua. Apna sawaal clearly bolein."
        : "Looks like there was some audio feedback. Please say your question clearly.";
      send(ws, { type: "assistant.start", emotion });
      send(ws, { type: "assistant.done", text: prompt, latencyMs: 35 });
      return;
    }
  }

  sessionManager.setEmotion(sessionId, emotion);
  sessionManager.appendTurn(sessionId, "user", contextResolvedText, { emotion });
  runtimeMetricsService.inc("utterances");
  const dialog = buildDialogDirective(session, contextResolvedText);
  sessionManager.setDialogState(sessionId, {
    currentTopics: dialog.mergedTopics,
    lastUserIntent: contextResolvedText,
    lastResolvedInstruction: dialog.directive
  });
  sessionManager.setDialogState(sessionId, {
    currentTopics: dialog.mergedTopics,
    lastUserIntent: cleanText,
    lastResolvedInstruction: dialog.directive
  });
  const summary = conversationMemoryService.buildSummary(session.turns);
  sessionManager.setConversationSummary(sessionId, summary);

  // ── Conversational handlers (instant, no LLM needed) ──────────────────────

  const q = cleanText.toLowerCase().trim();

  // "How are you" — check BEFORE greeting so "Hello how are you" works
  if (/how are you|how r u|how are u|aap kaise hain|kaise ho|kaisa hai|you okay|you good/i.test(q)) {
    const reply = isHindiMode
      ? "Main bilkul theek hoon, shukriya poochne ke liye! Aap batao, main aapki kaise help kar sakti hoon?"
      : "I'm doing great, thanks for asking! Now tell me, how can I help you today?";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // "Who are you / What is your name" — check BEFORE greeting so "Hello who are you" works
  if (/who are you|what is your name|your name|aap kaun hain|aapka naam|tumhara naam|naam kya hai|what's your name|whats your name|kon ho aap|kaun ho aap|aap kon hain|tumhara naam kya hai/i.test(q)) {
    const reply = isHindiMode
      ? "Main Aria hoon — BML Munjal University ki admissions counselor. Aap mujhse courses, fees, scholarship, hostel, ya admission ke baare mein kuch bhi pooch sakte hain!"
      : "I'm Aria, an admissions counselor at BML Munjal University. You can ask me anything about our courses, fees, scholarships, hostel, or the admission process!";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // "Are you an AI / Are you a robot" — check BEFORE greeting
  if (/\b(ai|robot|bot|artificial|machine|computer|human|real person|real human|chatbot)\b/i.test(q) &&
      /\b(are you|you a|you an|is this|am i talking)\b/i.test(q)) {
    const reply = isHindiMode
      ? "Nahi nahi, main Aria hoon — BMU ki admissions team mein hoon. Aap mujhse seedha baat kar rahe hain! Koi sawaal hai admission ke baare mein?"
      : "No no, I'm Aria — I'm part of the admissions team here at BMU. You're talking to a real person! Is there something about our programs I can help you with?";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // Pure greetings only (no extra words after)
  const isGreeting = /^(hi+|hello+|hey+|helo|hii+|namaste|namaskar|good morning|good afternoon|good evening|good day|howdy|greetings|नमस्ते|हेलो|हाय|haan|suno|allo|helo)\s*[!.]*$/i.test(q);
  if (isGreeting) {
    const replies = isHindiMode ? [
      "Namaste! Main Aria hoon, BMU ki admissions counselor. Aap kaise help kar sakti hoon aapki?",
      "Hello! Aria bol rahi hoon BMU se. Kya main aapki kuch help kar sakti hoon?",
      "Namaste! BMU mein aapka swagat hai. Main Aria hoon — courses, fees, scholarship, ya admission ke baare mein kuch jaanna hai?"
    ] : [
      "Hello! This is Aria from BML Munjal University admissions. How can I help you today?",
      "Hi there! Aria here from BMU. What can I help you with?",
      "Hello, welcome to BMU! I'm Aria from the admissions team. How can I assist you?"
    ];
    const reply = replies[Math.floor(Math.random() * replies.length)];
    sessionManager.appendTurn(sessionId, "assistant", reply, { greeting: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // Hindi/Hinglish greetings with questions embedded
  if (/^(hello|hi|hey|namaste|namaskar)\s+(kon|kaun|kya|aap|bhai|ji|sir|mam)/i.test(q) ||
      /^(kon|kaun)\s+(ho|hain|hai)\s*(aap|tum|you)/i.test(q) ||
      /^(aap|tum)\s+(kon|kaun)\s*(ho|hain|hai)/i.test(q)) {
    const reply = isHindiMode
      ? "Main Aria hoon — BML Munjal University ki admissions counselor. Aap mujhse courses, fees, scholarship, hostel, ya admission ke baare mein kuch bhi pooch sakte hain!"
      : "I'm Aria, an admissions counselor at BML Munjal University. You can ask me anything about our courses, fees, scholarships, hostel, or the admission process!";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // "Thank you" responses — check anywhere in the sentence
  if (/thank you|thanks|thank u|thankyou|shukriya|dhanyawad|thnx|thx/i.test(q) &&
      /done|bye|goodbye|nice|great|helpful|good|amazing|wonderful|speaking|guiding|helped/i.test(q)) {
    const reply = isHindiMode
      ? "Bahut shukriya! Aapka swagat hai BMU mein. Koi bhi sawaal ho toh zaroor poochein. All the best!"
      : "Thank you so much, it was lovely speaking with you! Best of luck with your admission. Feel free to call us anytime at plus 91 1275 286101. Take care!";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // Simple "thank you" responses
  if (/^(thank you|thanks|thank u|thankyou|shukriya|dhanyawad|bahut shukriya|thnx|thx)/i.test(q)) {
    const reply = isHindiMode
      ? "Koi baat nahi! Aur kuch jaanna hai BMU ke baare mein?"
      : "You're welcome! Is there anything else you'd like to know about BMU?";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // "Okay / Got it / Sure" acknowledgements
  if (/^(okay|ok|got it|alright|sure|fine|understood|theek hai|achha|haan|ji haan|bilkul|right|yep|yup|yeah|thik hai|sahi hai|samajh gaya|samajh gayi)/i.test(q) && q.length < 25) {
    const reply = isHindiMode
      ? "Haan, toh aur kuch poochna hai? Main yahan hoon help karne ke liye."
      : "Great! Feel free to ask anything else about BMU — I'm here to help.";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // "Bye / Goodbye" endings
  if (/^(bye|goodbye|good bye|see you|take care|alvida|phir milenge|ok bye|okay bye|tata)/i.test(q)) {
    const reply = isHindiMode
      ? "Theek hai, phir milenge! BMU mein aapka swagat hai. Koi bhi sawaal ho toh call karein — +91-1275-286101. All the best!"
      : "Take care! It was great talking to you. If you have any more questions, feel free to call us at plus 91 1275 286101. All the best with your admission!";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // "Can you help me / I need help"
  if (/^(can you help|help me|i need help|mujhe help chahiye|help chahiye|koi help|please help|can u help|help karo)/i.test(q)) {
    const reply = isHindiMode
      ? "Haan bilkul! Main yahan hoon aapki help ke liye. Courses, fees, scholarship, hostel, ya admission — kisi bhi cheez ke baare mein poochein."
      : "Of course! That's exactly what I'm here for. Ask me anything about BMU — courses, fees, scholarships, hostel, or admissions. What would you like to know?";
    sessionManager.appendTurn(sessionId, "assistant", reply, { smalltalk: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: reply, latencyMs: 20 });
    return;
  }

  // Step 9: TF-IDF retrieval
  const retrieved = tfidfRetrieve(contextResolvedText);

  // Step 9b: Sentence-BERT semantic search (Python) — enhances retrieval
  // Run in parallel, use results if TF-IDF confidence is low
  let semanticResults = null;
  if (pythonNlpService.isAvailable() && retrieved.confidence < 0.4) {
    semanticResults = await pythonNlpService.semanticSearch(contextResolvedText, 3).catch(() => null);
    if (semanticResults?.results?.length > 0) {
      // Inject semantic search results as additional retrieval snippets
      const semanticSnippets = semanticResults.results.map(r => ({
        text: r.text,
        source: `semantic/${r.id}`,
        score: r.score
      }));
      retrieved.snippets = [...retrieved.snippets, ...semanticSnippets].slice(0, 4);
      if (semanticResults.confidence > retrieved.confidence) {
        retrieved.confidence = semanticResults.confidence;
      }
    }
  }

  const policyGrounded = responsePolicy.resolveGroundedAnswer(contextResolvedText, emotion.languageMode);

  // Semantic similarity fallback — if policy misses, try paraphrase matching
  if (!policyGrounded && retrieved.snippets.length === 0) {
    const similar = findSimilarTopic(contextResolvedText);
    if (similar) {
      const similarGrounded = responsePolicy.resolveGroundedAnswer(similar.canonical, emotion.languageMode);
      if (similarGrounded) {
        runtimeMetricsService.inc("groundedAnswers");
        sessionManager.appendTurn(sessionId, "assistant", similarGrounded.text, {
          grounded: true, semantic: true, score: similar.score
        });
        send(ws, { type: "assistant.start", emotion });
        send(ws, { type: "assistant.done", text: similarGrounded.text, grounded: true, latencyMs: 40 });
        return;
      }
    }
  }

  // Always prefer grounded knowledge base answers — skip LLM entirely when confident
  if (policyGrounded && policyGrounded.confidence >= 0.55) {
    runtimeMetricsService.inc("groundedAnswers");
    sessionManager.appendTurn(sessionId, "assistant", policyGrounded.text, {
      grounded: true,
      source: policyGrounded.source,
      confidence: policyGrounded.confidence
    });
    send(ws, { type: "assistant.start", emotion });
    send(ws, {
      type: "assistant.done",
      text: policyGrounded.text,
      grounded: true,
      source: policyGrounded.source,
      confidence: policyGrounded.confidence,
      latencyMs: 35
    });
    return;
  }

  if (!policyGrounded && retrieved.snippets.length === 0) {
    const safeClarify = isHindiMode
      ? "यह एक अच्छा सवाल है! इसके सबसे accurate details के लिए हमारी admissions team से बात करें — +91-1275-286101 पर call करें या admissions@bmu.edu.in पर email करें। क्या BMU के courses, fees, या scholarship के बारे में कुछ और जानना है?"
      : "Great question! For the most accurate details on that, our admissions team would be best placed to help — call +91-1275-286101 or email admissions@bmu.edu.in. Is there anything about BMU courses, fees, or scholarships I can help with?";
    sessionManager.appendTurn(sessionId, "assistant", safeClarify, { clarification: true });
    send(ws, { type: "assistant.start", emotion });
    send(ws, { type: "assistant.done", text: safeClarify, latencyMs: 40 });
    return;
  }

  send(ws, { type: "assistant.start", emotion });
  send(ws, { type: "assistant.ack", text: buildRealtimeAck(isHindiMode, correctedText) });
  const startedAt = Date.now();
  runtimeMetricsService.inc("llmCalls");
  const abortController = llmService.createAbortController();
  const generationSeq = sessionManager.startGeneration(sessionId, abortController);
  let lastTokenAt = Date.now();
  let firstTokenAt = 0;

  const progressTimer = setInterval(() => {
    if (!sessionManager.isGenerationActive(sessionId, generationSeq)) return;
    send(ws, { type: "assistant.progress", elapsedMs: Date.now() - startedAt });
  }, config.ackIntervalMs);

  const silenceWatchdog = setInterval(() => {
    if (!sessionManager.isGenerationActive(sessionId, generationSeq)) return;
    if (Date.now() - lastTokenAt > config.llmMaxSilenceMs) {
      try { abortController.abort(); } catch { /* ignore */ }
    }
  }, 500);

  let assembled = "";
  try {
    const finalText = await llmService.streamReply(
      {
        turns: session.turns,
        userText: contextResolvedText,
        emotion,
        languageMode: emotion.languageMode,
        conversationSummary: session.conversationSummary,
        retrievalSnippets: retrieved.snippets,
        dialogDirective: dialog.directive,
        entityContext
      },
      (token) => {
        if (!sessionManager.isGenerationActive(sessionId, generationSeq)) return;
        if (!firstTokenAt) firstTokenAt = Date.now();
        lastTokenAt = Date.now();
        assembled += token;
        send(ws, { type: "assistant.chunk", token });
      },
      abortController.signal
    );

    if (!sessionManager.isGenerationActive(sessionId, generationSeq)) return;
    const rawResult = finalText || assembled || "I am here with you. Please continue.";

    // Hallucination detection — check if LLM contradicts known facts
    const halluCheck = detectHallucination(rawResult);
    if (halluCheck.isHallucination) {
      console.warn("[Hallucination detected]", halluCheck.issues);
      // Fall back to grounded answer if available
      const groundedFallback = responsePolicy.resolveGroundedAnswer(contextResolvedText, emotion.languageMode);
      if (groundedFallback) {
        sessionManager.appendTurn(sessionId, "assistant", groundedFallback.text, { hallucination_corrected: true });
        runtimeMetricsService.observeLatency(Date.now() - startedAt);
        send(ws, { type: "assistant.done", text: groundedFallback.text, latencyMs: Date.now() - startedAt });
        return;
      }
    }

    const result = llmService.postProcessReply({
      userText: contextResolvedText,
      replyText: rawResult,
      languageMode: emotion.languageMode,
      retrievalConfidence: retrieved.confidence,
      dialogDirective: dialog.directive
    });
    const finalResult = responseQualityService.enforce(result, {
      retrievalConfidence: retrieved.confidence,
      isHindiMode,
      strictMode: config.strictDemoMode
    });

    sessionManager.appendTurn(sessionId, "assistant", finalResult, { latencyMs: Date.now() - startedAt });
    runtimeMetricsService.observeLatency(Date.now() - startedAt);

    send(ws, {
      type: "assistant.done",
      text: finalResult,
      sources: retrieved.snippets.slice(0, 3).map((s) => ({ source: s.source, score: Number(s.score.toFixed(2)) })),
      retrievalConfidence: retrieved.confidence,
      timeToFirstTokenMs: firstTokenAt ? firstTokenAt - startedAt : 0,
      latencyMs: Date.now() - startedAt
    });
  } catch (error) {
    const errorText = String(error);
    if (errorText.includes("generation_aborted")) return;
    runtimeMetricsService.inc("llmErrors");
    const fallback = isHindiMode
      ? "यह एक अच्छा सवाल है! इसके सबसे accurate details के लिए हमारी admissions team से बात करें — +91-1275-286101 पर call करें या admissions@bmu.edu.in पर email करें।"
      : "Great question! For the most accurate details, our admissions team can help — call +91-1275-286101 or email admissions@bmu.edu.in.";
    sessionManager.appendTurn(sessionId, "assistant", fallback, { error: errorText });
    send(ws, { type: "assistant.done", text: fallback, error: false });
  } finally {
    clearInterval(progressTimer);
    clearInterval(silenceWatchdog);
    sessionManager.finishGeneration(sessionId, generationSeq);
  }
}
