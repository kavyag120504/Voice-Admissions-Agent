const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const interruptBtn = document.getElementById("interruptBtn");
const sendBtn = document.getElementById("sendBtn");
const manualInput = document.getElementById("manualInput");
const languageMode = document.getElementById("languageMode");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

// ── State ────────────────────────────────────────────────────────────────────
let ws = null;
let sessionId = null;
let recognition = null;
let isRunning = false;
let finalAssistantText = "";
let speaking = false;
let assistantThinking = false;
let streamSpeakBuffer = "";
let speechQueue = [];
let currentUtterance = null;       // BufferSource (Piper) or SpeechSynthesisUtterance
let activeAssistantEmotion = "neutral";
let lastInterruptAt = 0;
let recentAssistantSpeechText = "";
let assistantSpeechStartedAt = 0;
let lastSentUserText = "";
let lastSentAt = 0;
let pendingSpeakTimer = null;
let agentState = "idle";
let voicesReady = false;
let piperAvailable = false;
let isSpeakingNext = false;        // guard against concurrent speakNext calls

const AGENT_STATES = { IDLE: "idle", LISTENING: "listening", THINKING: "thinking", SPEAKING: "speaking" };

// ── Logging ──────────────────────────────────────────────────────────────────
function log(line) {
  logEl.textContent += `${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}
function setStatus(text) { statusEl.textContent = `Status: ${text}`; }
function setAgentState(nextState, detail = "") {
  agentState = nextState;
  setStatus(detail ? `${nextState} ${detail}` : nextState);
}

// ── Text helpers ─────────────────────────────────────────────────────────────
function normalizeForCompare(text) {
  return String(text || "").toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim();
}

let lastSpeechEndedAt = 0; // track when Aria last finished speaking
let lastSpeechStartedAt = 0; // track when Aria started speaking

function isLikelyEchoFromAssistant(userText) {
  // Block for 4 seconds after Aria finishes speaking — mic echo window
  const timeSinceSpeechEnded = Date.now() - lastSpeechEndedAt;
  if (timeSinceSpeechEnded < 4000) return true;

  // Block while Aria is actively speaking
  if (speaking) return true;

  // Check word overlap with recent assistant speech (catches delayed echo)
  const u = normalizeForCompare(userText);
  if (u.length < 5) return false;
  const a = normalizeForCompare(recentAssistantSpeechText.slice(-200));
  if (a.length < 10) return false;
  const uWords = u.split(" ").filter(w => w.length > 2);
  const aWords = new Set(a.split(" ").filter(w => w.length > 2));
  let overlap = 0;
  for (const w of uWords) if (aWords.has(w)) overlap++;
  // If more than 40% of user words match assistant's recent speech → echo
  return uWords.length > 0 && overlap / uWords.length >= 0.4;
}

function isMeaningfulSpeechText(text) {
  const t = String(text || "").trim();
  if (t.length < 3) return false;
  if (!/[\p{L}\p{N}]/u.test(t)) return false;
  return normalizeForCompare(t).split(" ").filter(Boolean).length >= 2 || t.length >= 8;
}

function extractSpeakableSentences(buffer) {
  const match = buffer.match(/(.+?[.!?।])(\s|$)/g);
  if (!match) return { sentences: [], rest: buffer };
  const consumed = match.join("");
  return { sentences: match.map(s => s.trim()).filter(Boolean), rest: buffer.slice(consumed.length) };
}

function selectVoiceLang(text) {
  return /[\u0900-\u097F]/.test(text) ? "hi-IN" : "en-IN";
}

// ── Language lock ─────────────────────────────────────────────────────────────
let lockedLang = null; // null = not locked yet, locks after first utterance

function detectLangFromText(text) {
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  const hasEnglish = /[a-zA-Z]/.test(text);
  if (hasDevanagari && !hasEnglish) return "hi-IN";
  if (hasEnglish && !hasDevanagari) return "en-IN";
  // Hinglish — check which is dominant
  const hindiWords = (text.match(/[\u0900-\u097F]+/g) || []).length;
  const engWords = (text.match(/[a-zA-Z]+/g) || []).length;
  return hindiWords > engWords ? "hi-IN" : "en-IN";
}

function getRecognitionLang() {
  if (lockedLang) return lockedLang;
  const sel = languageMode.value;
  if (sel === "hi-IN") return "hi-IN";
  if (sel === "en-IN") return "en-IN";
  return "en-IN"; // default for auto
}
const VOICE_PRIORITY = {
  "hi":    ["Google हिन्दी", "Microsoft Heera"],
  "en-IN": ["Microsoft Heera", "Microsoft Ravi", "Google UK English Female"],
  "en":    ["Microsoft Heera", "Google UK English Female", "Microsoft Zira", "Microsoft Hazel", "Google US English"]
};

function chooseBestVoice(langCode) {
  const voices = window.speechSynthesis?.getVoices?.() || [];
  if (!voices.length) return null;
  const isHindi = langCode === "hi-IN" || langCode === "hi";
  const key = isHindi ? "hi" : langCode.startsWith("en-IN") ? "en-IN" : "en";
  for (const name of (VOICE_PRIORITY[key] || VOICE_PRIORITY["en"])) {
    const v = voices.find(v => v.name === name);
    if (v) return v;
  }
  return voices.find(v => v.lang === "en-IN") || voices.find(v => v.lang.startsWith("en")) || voices[0];
}

function initVoices() {
  if (!window.speechSynthesis) return;
  const ready = window.speechSynthesis.getVoices();
  if (ready?.length) voicesReady = true;
  window.speechSynthesis.onvoiceschanged = () => { voicesReady = true; };
}

// ── Piper TTS ─────────────────────────────────────────────────────────────────
async function checkPiperStatus() {
  try {
    const res = await fetch("/api/tts/status");
    const data = await res.json();
    piperAvailable = !!data.available;
    console.log("Piper TTS available:", piperAvailable);
  } catch { piperAvailable = false; }
}

async function speakWithPiper(text) {
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, emotion: activeAssistantEmotion })
    });
    if (!res.ok) throw new Error("piper_http_error");
    const arrayBuffer = await res.arrayBuffer();
    if (!arrayBuffer.byteLength) throw new Error("empty_audio");

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);

    speaking = true;
    assistantSpeechStartedAt = Date.now();
    lastSpeechStartedAt = Date.now();
    currentUtterance = source;
    // Stop mic while Aria is speaking to prevent echo
    if (recognition) { try { recognition.stop(); } catch (_) {} }
    setAgentState(AGENT_STATES.SPEAKING);

    return new Promise(resolve => {
      source.onended = () => {
        speaking = false;
        currentUtterance = null;
        lastSpeechEndedAt = Date.now();
        recentAssistantSpeechText = `${recentAssistantSpeechText} ${text}`.trim().slice(-800);
        // Wait 3 seconds then restart mic
        setTimeout(() => {
          isSpeakingNext = false;
          speakNext();
          // Restart mic after echo window
          if (isRunning && recognition && speechQueue.length === 0) {
            setTimeout(() => {
              if (isRunning && !speaking) {
                try { recognition.start(); } catch (_) {}
              }
            }, 1000);
          }
        }, 500);
        resolve(true);
      };
      source.start(0);
    });
  } catch (e) {
    console.warn("Piper failed:", e.message, "— falling back to browser TTS");
    return false;
  }
}

// ── Speech queue ──────────────────────────────────────────────────────────────
function enqueueSpeech(text) {
  const clean = String(text || "").trim();
  if (!clean) return;
  speechQueue.push(clean);
  recentAssistantSpeechText = `${recentAssistantSpeechText} ${clean}`.trim().slice(-600);
  if (!speaking && !isSpeakingNext) speakNext();
}

async function speakNext() {
  if (isSpeakingNext || speaking) return;
  isSpeakingNext = true;

  try {
    const next = speechQueue.shift();
    if (!next) {
      if (isRunning) setAgentState(AGENT_STATES.LISTENING);
      return;
    }

    // Try Piper first
    if (piperAvailable) {
      const ok = await speakWithPiper(next);
      if (ok) return; // speakNext called again from onended
    }

    // Browser TTS fallback
    if (!window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(next);
    const lang = selectVoiceLang(next);
    utter.lang = lang;
    const voice = chooseBestVoice(lang);
    if (voice) utter.voice = voice;

    // Adjust voice tone based on detected user emotion
    switch (activeAssistantEmotion) {
      case "happy":
        utter.rate = 1.0;    // slightly faster, energetic
        utter.pitch = 1.1;   // higher pitch, warm
        break;
      case "stressed":
        utter.rate = 0.85;   // slower, calming
        utter.pitch = 0.95;  // slightly lower, reassuring
        break;
      case "sad":
        utter.rate = 0.82;   // slow, gentle
        utter.pitch = 0.92;  // softer tone
        break;
      case "angry":
        utter.rate = 0.88;   // measured, calm
        utter.pitch = 0.95;  // neutral, de-escalating
        break;
      default:
        utter.rate = 0.92;
        utter.pitch = 1.0;
    }
    utter.volume = 1.0;

    utter.onstart = () => {
      currentUtterance = utter;
      speaking = true;
      assistantSpeechStartedAt = Date.now();
      setAgentState(AGENT_STATES.SPEAKING);
    };
    utter.onend = () => {
      speaking = false;
      currentUtterance = null;
      lastSpeechEndedAt = Date.now();
      isSpeakingNext = false;
      speakNext();
    };
    utter.onerror = () => {
      speaking = false;
      currentUtterance = null;
      isSpeakingNext = false;
      speakNext();
    };
    window.speechSynthesis.speak(utter);
  } finally {
    // Only release lock if we didn't hand off to Piper (Piper releases it via onended)
    if (!speaking) isSpeakingNext = false;
  }
}

// ── Stop all audio ────────────────────────────────────────────────────────────
function stopAllAudio() {
  if (currentUtterance && typeof currentUtterance.stop === "function") {
    try { currentUtterance.stop(); } catch (_) {}
  }
  window.speechSynthesis?.cancel();
  speaking = false;
  currentUtterance = null;
  speechQueue = [];
  streamSpeakBuffer = "";
  isSpeakingNext = false;
}

// ── Interrupt ─────────────────────────────────────────────────────────────────
function interruptAssistant(reason = "user") {
  if (!ws || ws.readyState !== WebSocket.OPEN || !sessionId) return;
  const now = Date.now();
  if (now - lastInterruptAt < 450) return;
  lastInterruptAt = now;

  stopAllAudio();
  assistantThinking = false;
  if (pendingSpeakTimer) { clearTimeout(pendingSpeakTimer); pendingSpeakTimer = null; }
  if (isRunning) setAgentState(AGENT_STATES.LISTENING);
  ws.send(JSON.stringify({ type: "session.interrupt", sessionId, reason }));
}

// ── ASR repair ────────────────────────────────────────────────────────────────
async function repairTranscript(text, confidence = 1) {
  try {
    const res = await fetch("/api/asr/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, confidence })
    });
    if (!res.ok) return text;
    return (await res.json()).text || text;
  } catch { return text; }
}

// ── Speech recognition ────────────────────────────────────────────────────────
function setupRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) { alert("SpeechRecognition not supported in this browser. Use Chrome."); return null; }

  const rec = new Recognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = getRecognitionLang();

  rec.onresult = async (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const rawText = result[0]?.transcript?.trim();
      if (!rawText) continue;

      if (!result.isFinal) {
        // Track interim text for Done button
        lastInterimText = rawText;

        // Reset silence timer — auto-send after 3s of no new speech
        if (silenceTimer) clearTimeout(silenceTimer);
        if (!speaking && rawText.trim().length > 3) {
          silenceTimer = setTimeout(() => {
            if (lastInterimText && lastInterimText.trim().length > 2 && !speaking && isRunning) {
              log(`You: ${lastInterimText}`);
              sendUserUtterance(lastInterimText);
              lastInterimText = "";
            }
          }, 3000);
        }

        // Check for explicit interrupt words first — these always work
        const isInterruptWord = /^(wait|stop|ruko|bas|ek second|hold on|excuse me|sorry|suno)\s*[!.]*$/i.test(rawText.trim());

        if (speaking) {
          // While Piper is speaking — only interrupt on explicit words
          if (isInterruptWord) {
            interruptAssistant("barge-in-speaking");
          }
          // Never process echo as barge-in
          continue;
        }

        if (assistantThinking) {
          const wordCount = normalizeForCompare(rawText).split(" ").filter(Boolean).length;
          if (isInterruptWord || (rawText.length >= 6 && wordCount >= 1 &&
              Date.now() - assistantSpeechStartedAt > 500)) {
            interruptAssistant("barge-in-interim");
          }
        }
        continue;
      }

      // Final result — strict echo check
      if (speaking) continue;
      if (isLikelyEchoFromAssistant(rawText)) {
        console.log("Echo blocked:", rawText.slice(0, 40));
        continue;
      }

      const confidence = Number(result[0]?.confidence ?? 1);
      const text = await repairTranscript(rawText, confidence);
      if (!text || !isMeaningfulSpeechText(text) || isLikelyEchoFromAssistant(text)) continue;

      // Lock language after first real utterance (auto mode only)
      if (!lockedLang && languageMode.value === "auto") {
        lockedLang = detectLangFromText(text);
        rec.lang = lockedLang;
        console.log("Language locked to:", lockedLang);
      }

      if (assistantThinking) interruptAssistant("barge-in-final");
      log(`You: ${text}`);
      sendUserUtterance(text);
    }
  };

  rec.onerror = (e) => {
    if (e.error !== "no-speech") log(`Speech error: ${e.error}`);
  };

  rec.onend = () => {
    if (isRunning) {
      // Only restart mic if Aria is NOT speaking and echo window has passed
      const timeSinceSpeech = Date.now() - lastSpeechEndedAt;
      if (!speaking && !assistantThinking && timeSinceSpeech > 3000) {
        try { rec.start(); } catch (_) {}
      } else {
        // Retry after delay
        setTimeout(() => {
          if (isRunning && !speaking) {
            try { rec.start(); } catch (_) {}
          }
        }, Math.max(500, 3000 - (Date.now() - lastSpeechEndedAt)));
      }
    }
  };

  return rec;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function sendUserUtterance(text) {
  if (!ws || ws.readyState !== WebSocket.OPEN || !sessionId) return;
  const clean = String(text || "").trim();
  if (!clean) return;
  const now = Date.now();
  if (normalizeForCompare(clean) === normalizeForCompare(lastSentUserText) && now - lastSentAt < 2200) return;
  lastSentUserText = clean;
  lastSentAt = now;
  ws.send(JSON.stringify({ type: "user.utterance", sessionId, text: clean }));
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    const token = localStorage.getItem("bmu_ws_token") || "";
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${protocol}://${location.host}/realtime${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => ws.send(JSON.stringify({ type: "session.start" }));

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "session.started") {
        sessionId = msg.sessionId;
        setAgentState(AGENT_STATES.LISTENING);
        resolve();
        return;
      }

      if (msg.type === "assistant.start") {
        finalAssistantText = "";
        streamSpeakBuffer = "";
        assistantThinking = true;
        activeAssistantEmotion = msg.emotion?.emotion || "neutral";
        log(`Assistant[${activeAssistantEmotion}]: `);
        setAgentState(AGENT_STATES.THINKING);
        return;
      }
      if (msg.type === "assistant.ack") {
        if (msg.text) {
          if (pendingSpeakTimer) clearTimeout(pendingSpeakTimer);
          pendingSpeakTimer = setTimeout(() => { enqueueSpeech(msg.text); pendingSpeakTimer = null; }, 120);
        }
        return;
      }

      if (msg.type === "assistant.progress") {
        if (assistantThinking) {
          const sec = Math.max(1, Math.round(Number(msg.elapsedMs || 0) / 1000));
          setAgentState(AGENT_STATES.THINKING, `(${sec}s)`);
        }
        return;
      }

      if (msg.type === "assistant.chunk") {
        finalAssistantText += msg.token;
        streamSpeakBuffer += msg.token;
        const { sentences, rest } = extractSpeakableSentences(streamSpeakBuffer);
        streamSpeakBuffer = rest;
        sentences.forEach(s => enqueueSpeech(s));
        return;
      }

      if (msg.type === "assistant.done") {
        assistantThinking = false;
        if (pendingSpeakTimer) { clearTimeout(pendingSpeakTimer); pendingSpeakTimer = null; }
        if (!finalAssistantText.trim()) finalAssistantText = String(msg.text || "");
        log(`  ${msg.text}`);
        if (msg.grounded && msg.source) log(`  [Source] ${msg.source.title} | confidence=${msg.confidence ?? "n/a"}`);

        // Speak remaining buffer
        if (streamSpeakBuffer.trim()) { enqueueSpeech(streamSpeakBuffer.trim()); streamSpeakBuffer = ""; }
        // Force speak if nothing queued yet
        if (!speaking && speechQueue.length === 0 && finalAssistantText.trim()) {
          enqueueSpeech(finalAssistantText.trim());
        }
        return;
      }

      if (msg.type === "assistant.interrupted") {
        assistantThinking = false;
        speaking = false;
        setAgentState(AGENT_STATES.LISTENING);
        log(`System: ${msg.message || "Interrupted."}`);
        return;
      }

      if (msg.type === "error") log(`Error: ${msg.message}`);
    };

    ws.onerror = (e) => { console.error("WS error", e); reject(new Error("WebSocket connection failed")); };
    ws.onclose = () => { if (isRunning) setAgentState(AGENT_STATES.IDLE, "(disconnected)"); };
  });
}

// ── Button handlers ───────────────────────────────────────────────────────────
startBtn.onclick = async () => {
  if (isRunning) return;
  isRunning = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setAgentState(AGENT_STATES.THINKING, "(starting)");

  try {
    initVoices();
    await checkPiperStatus();
    await connectSocket();

    // Reset state
    recentAssistantSpeechText = "";
    finalAssistantText = "";
    lastSentUserText = "";
    lastSentAt = 0;
    speechQueue = [];
    streamSpeakBuffer = "";
    speaking = false;
    isSpeakingNext = false;
    lockedLang = null; // reset language lock for new call
    lastSpeechEndedAt = 0;

    recognition = setupRecognition();
    if (recognition) recognition.start();

    sendBtn.disabled = false;
    interruptBtn.disabled = false;
  } catch (error) {
    log(`Startup failed: ${error.message}`);
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    sendBtn.disabled = true;
    interruptBtn.disabled = true;
    setAgentState(AGENT_STATES.IDLE);
  }
};

stopBtn.onclick = () => {
  isRunning = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  sendBtn.disabled = true;
  interruptBtn.disabled = true;

  if (recognition) { try { recognition.stop(); } catch (_) {} recognition = null; }
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();

  stopAllAudio();
  assistantThinking = false;
  if (pendingSpeakTimer) { clearTimeout(pendingSpeakTimer); pendingSpeakTimer = null; }
  setAgentState(AGENT_STATES.IDLE);
  log("Call stopped.");
};

sendBtn.onclick = () => {
  const text = manualInput.value.trim();
  if (!text) return;
  if (speaking || assistantThinking) interruptAssistant("manual-text");
  log(`You(text): ${text}`);
  sendUserUtterance(text);
  manualInput.value = "";
};

interruptBtn.onclick = () => interruptAssistant("manual-interrupt");

manualInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendBtn.click(); });

let lastInterimText = "";
let silenceTimer = null;

window.addEventListener("user-done-speaking", () => {
  if (!isRunning || !sessionId) return;
  if (lastInterimText && lastInterimText.trim().length > 2) {
    if (assistantThinking || speaking) interruptAssistant("done-speaking");
    log(`You: ${lastInterimText}`);
    sendUserUtterance(lastInterimText);
    lastInterimText = "";
  }
});
