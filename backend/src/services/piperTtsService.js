import { spawn } from "child_process";
import { existsSync } from "fs";

const PIPER_EXE = "D:\\piper_windows_amd64\\piper\\piper.exe";
const PIPER_MODEL = "D:\\piper_windows_amd64\\piper\\en_US-amy-medium.onnx";

// Emotion → Piper speech rate (length_scale: higher = slower, lower = faster)
// 1.0 is normal speed
const EMOTION_SPEECH_RATE = {
  happy:   "0.85",  // faster, energetic
  neutral: "1.0",   // normal
  stressed:"1.15",  // slower, calming
  sad:     "1.2",   // slow, gentle
  angry:   "1.1"    // measured, calm
};

/**
 * Pre-process text to add natural human-like pauses and rhythm
 * Makes TTS sound more natural by adding strategic pauses
 */
function humanizeText(text, emotion) {
  let t = String(text || "")
    .replace(/[₹*_#`]/g, "")   // remove markdown
    .replace(/\s+/g, " ")
    .trim();

  // Add pause after filler words (they naturally pause after these)
  t = t.replace(/\b(So|Honestly|Actually|Look|See|Right|Oh|Well|Now)\b,?\s/g, "$1, ");

  // Add slight pause before numbers (humans pause before stating numbers)
  t = t.replace(/(\s)(₹?\d+\.?\d*\s*(lakh|lakhs|percent|%|LPA|lpa|year|years))/gi, "$1... $2");

  // Add pause after question acknowledgement
  t = t.replace(/\b(Great|Good|Perfect|Excellent|Absolutely|Sure|Of course)\b[,.]?\s/g, "$1. ");

  // Stressed emotion — add more pauses, slower feel
  if (emotion === "stressed" || emotion === "sad") {
    t = t.replace(/\. /g, "... ");  // longer pauses between sentences
    t = t.replace(/Don't worry/gi, "Don't worry at all");
  }

  // Happy emotion — make it flow faster, less pauses
  if (emotion === "happy") {
    t = t.replace(/\.\.\./g, ".");
  }

  return t;
}

export const piperTtsService = {
  isAvailable() {
    return existsSync(PIPER_EXE) && existsSync(PIPER_MODEL);
  },

  /**
   * Convert text to WAV audio buffer using Piper
   * emotion param adjusts speech rate for human-like delivery
   */
  synthesize(text, emotion = "neutral") {
    return new Promise((resolve, reject) => {
      if (!this.isAvailable()) {
        return reject(new Error("piper_not_available"));
      }

      const processedText = humanizeText(text, emotion);
      if (!processedText) return reject(new Error("empty_text"));

      const lengthScale = EMOTION_SPEECH_RATE[emotion] || "1.0";

      const chunks = [];
      const proc = spawn(PIPER_EXE, [
        "--model", PIPER_MODEL,
        "--output-raw",
        "--length-scale", lengthScale
      ]);

      proc.stdout.on("data", chunk => chunks.push(chunk));
      proc.stderr.on("data", () => {});

      proc.on("close", code => {
        if (code !== 0 && chunks.length === 0) {
          return reject(new Error(`piper_exit_${code}`));
        }
        const pcm = Buffer.concat(chunks);
        const wav = pcmToWav(pcm, 22050, 1, 16);
        resolve(wav);
      });

      proc.on("error", err => reject(err));

      proc.stdin.write(processedText);
      proc.stdin.end();
    });
  }
};

function pcmToWav(pcmBuffer, sampleRate = 22050, channels = 1, bitDepth = 16) {
  const byteRate = sampleRate * channels * (bitDepth / 8);
  const blockAlign = channels * (bitDepth / 8);
  const dataSize = pcmBuffer.length;
  const headerSize = 44;
  const buffer = Buffer.alloc(headerSize + dataSize);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(channels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitDepth, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  pcmBuffer.copy(buffer, 44);

  return buffer;
}
