import cors from "cors";
import express from "express";
import { WebSocketServer } from "ws";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "./config.js";
import { handleRealtimeMessage } from "./realtime/pipeline.js";
import { asrFallbackService } from "./services/asrFallbackService.js";
import { runtimeMetricsService } from "./services/runtimeMetricsService.js";
import { piperTtsService } from "./services/piperTtsService.js";

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(cors({ origin: config.allowOrigin }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicPath = path.join(__dirname, "../public");
app.use(express.static(publicPath));

app.get("/health", (_req, res) => {
  res.json({ ok: true, mode: "realtime-web-call", model: config.ollamaModel });
});

app.post("/api/asr/repair", (req, res) => {
  const text = String(req.body?.text || "");
  const confidence = Number(req.body?.confidence ?? 1);
  const repaired = asrFallbackService.repairTranscript(text, confidence);
  res.json({ ok: true, ...repaired });
});

// Piper TTS endpoint — returns WAV audio
app.post("/api/tts", async (req, res) => {
  const text = String(req.body?.text || "").trim();
  const emotion = String(req.body?.emotion || "neutral");
  if (!text) return res.status(400).json({ error: "no text" });
  try {
    const wav = await piperTtsService.synthesize(text, emotion);
    res.set("Content-Type", "audio/wav");
    res.set("Content-Length", wav.length);
    res.send(wav);
  } catch (err) {
    res.status(500).json({ error: String(err.message) });
  }
});

app.get("/api/tts/status", (_req, res) => {
  res.json({ available: piperTtsService.isAvailable() });
});

app.get("/metrics", (_req, res) => {
  res.json({ ok: true, ...runtimeMetricsService.snapshot() });
});

const server = createServer(app);
const wss = new WebSocketServer({ server, path: "/realtime" });

wss.on("connection", (ws, request) => {
  if (config.wsAuthToken) {
    const url = new URL(request.url || "", `http://${request.headers.host}`);
    const token = url.searchParams.get("token");
    if (token !== config.wsAuthToken) {
      ws.close(1008, "Unauthorized websocket token");
      return;
    }
  }

  ws.isAlive = true;
  ws.on("pong", () => { ws.isAlive = true; });

  ws.on("message", async (buffer) => {
    try {
      const msg = JSON.parse(buffer.toString("utf-8"));
      await handleRealtimeMessage(ws, msg);
    } catch (_e) {
      ws.send(JSON.stringify({ type: "error", message: "Invalid message payload." }));
    }
  });
});

const heartbeatInterval = setInterval(() => {
  wss.clients.forEach((client) => {
    if (!client.isAlive) { client.terminate(); return; }
    client.isAlive = false;
    client.ping();
  });
}, 30000);

wss.on("close", () => clearInterval(heartbeatInterval));

server.listen(config.port, () => {
  console.log(`BMU advanced call agent running at http://localhost:${config.port}`);
  console.log(`Piper TTS available: ${piperTtsService.isAvailable()}`);
});
