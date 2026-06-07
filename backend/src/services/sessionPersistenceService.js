import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const storagePath = path.join(__dirname, "../../data/session_store.json");

function sanitizeSession(session) {
  return {
    sessionId: session.sessionId,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    turns: Array.isArray(session.turns) ? session.turns.slice(-200) : [],
    lastEmotion: session.lastEmotion || { emotion: "neutral", confidence: 0.5, languageMode: "english" },
    inFlight: false,
    conversationSummary: session.conversationSummary || "",
    generationSeq: Number(session.generationSeq || 0),
    generationController: null
  };
}

export const sessionPersistenceService = {
  loadSessions() {
    try {
      if (!fs.existsSync(storagePath)) return [];
      const raw = fs.readFileSync(storagePath, "utf-8");
      const parsed = JSON.parse(raw);
      const sessions = Array.isArray(parsed.sessions) ? parsed.sessions : [];
      return sessions.map(sanitizeSession);
    } catch {
      return [];
    }
  },
  saveSessions(sessionsArray) {
    try {
      const payload = {
        savedAt: Date.now(),
        sessions: sessionsArray.map(sanitizeSession)
      };
      fs.writeFileSync(storagePath, JSON.stringify(payload, null, 2), "utf-8");
      return true;
    } catch {
      return false;
    }
  }
};
