import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultGlobalDataDir = path.resolve(__dirname, "../../../data");

export const config = {
  port: Number(process.env.PORT || 3210),
  allowOrigin: process.env.ALLOW_ORIGIN || "http://localhost:3210",
  ollamaBaseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
  ollamaModel: process.env.OLLAMA_MODEL || "llama3.1:8b",
  maxContextTurns: Number(process.env.MAX_CONTEXT_TURNS || 40),
  wsAuthToken: process.env.WS_AUTH_TOKEN || "",
  maxInputChars: Number(process.env.MAX_INPUT_CHARS || 420),
  requestTimeoutMs: Number(process.env.REQUEST_TIMEOUT_MS || 18000),
  globalDataDir: process.env.BMU_GLOBAL_DATA_DIR || defaultGlobalDataDir,
  retrievalTopK: Number(process.env.RETRIEVAL_TOP_K || 4),
  minRetrievalConfidence: Number(process.env.MIN_RETRIEVAL_CONFIDENCE || 0.22),
  strictDemoMode: String(process.env.STRICT_DEMO_MODE || "true").toLowerCase() === "true",
  maxResponseSentences: Number(process.env.MAX_RESPONSE_SENTENCES || 4),
  llmMaxSilenceMs: Number(process.env.LLM_MAX_SILENCE_MS || 9000),
  ackIntervalMs: Number(process.env.ACK_INTERVAL_MS || 1400),
  preferredModels: String(
    process.env.PREFERRED_OLLAMA_MODELS ||
      "phi3.5,phi3:mini,llama3.1:8b,llama3.2:3b,bmu-aria:latest"
  )
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
};
