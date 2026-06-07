import crypto from "crypto";
import { sessionPersistenceService } from "../services/sessionPersistenceService.js";

class SessionManager {
  constructor() {
    this.sessions = new Map();
    this.persistTimer = null;
    this.bootstrap();
  }

  bootstrap() {
    const loaded = sessionPersistenceService.loadSessions();
    loaded.forEach((session) => {
      this.sessions.set(session.sessionId, session);
    });
  }

  schedulePersist() {
    if (this.persistTimer) clearTimeout(this.persistTimer);
    this.persistTimer = setTimeout(() => {
      this.flushPersist();
    }, 250);
  }

  flushPersist() {
    const array = Array.from(this.sessions.values());
    sessionPersistenceService.saveSessions(array);
    this.persistTimer = null;
  }

  createSession() {
    const sessionId = crypto.randomUUID();
    const state = {
      sessionId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      turns: [],
      lastEmotion: { emotion: "neutral", confidence: 0.5, languageMode: "english" },
      inFlight: false,
      conversationSummary: "",
      generationSeq: 0,
      generationController: null,
      dialogState: {
        currentTopics: [],
        lastUserIntent: "",
        lastResolvedInstruction: ""
      }
    };
    this.sessions.set(sessionId, state);
    this.schedulePersist();
    return state;
  }

  getSession(sessionId) {
    return this.sessions.get(sessionId);
  }

  appendTurn(sessionId, role, content, meta = {}) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    session.turns.push({ role, content, meta, ts: Date.now() });
    session.updatedAt = Date.now();
    if (session.turns.length > 200) session.turns = session.turns.slice(-200);
    this.schedulePersist();
    return session;
  }

  setEmotion(sessionId, emotionState) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    session.lastEmotion = emotionState;
    session.updatedAt = Date.now();
    this.schedulePersist();
    return session;
  }

  setInFlight(sessionId, inFlight) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    session.inFlight = inFlight;
    session.updatedAt = Date.now();
    this.schedulePersist();
    return session;
  }

  setConversationSummary(sessionId, summary) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    session.conversationSummary = summary || "";
    session.updatedAt = Date.now();
    this.schedulePersist();
    return session;
  }

  startGeneration(sessionId, controller) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    session.generationSeq += 1;
    session.generationController = controller || null;
    session.inFlight = true;
    session.updatedAt = Date.now();
    this.schedulePersist();
    return session.generationSeq;
  }

  isGenerationActive(sessionId, seq) {
    const session = this.sessions.get(sessionId);
    if (!session) return false;
    return session.inFlight && session.generationSeq === seq;
  }

  finishGeneration(sessionId, seq) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    if (session.generationSeq === seq) {
      session.inFlight = false;
      session.generationController = null;
      session.updatedAt = Date.now();
      this.schedulePersist();
    }
    return session;
  }

  cancelGeneration(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session || !session.inFlight) return false;
    try {
      session.generationController?.abort();
    } catch {
      // ignore abort failures
    }
    session.generationSeq += 1;
    session.generationController = null;
    session.inFlight = false;
    session.updatedAt = Date.now();
    this.schedulePersist();
    return true;
  }

  setDialogState(sessionId, partialState = {}) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;
    const prev = session.dialogState || {
      currentTopics: [],
      lastUserIntent: "",
      lastResolvedInstruction: ""
    };
    session.dialogState = {
      ...prev,
      ...partialState
    };
    session.updatedAt = Date.now();
    this.schedulePersist();
    return session.dialogState;
  }
}

export const sessionManager = new SessionManager();
