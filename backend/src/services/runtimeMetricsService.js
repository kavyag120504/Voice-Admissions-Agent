class RuntimeMetricsService {
  constructor() {
    this.startedAt = Date.now();
    this.counters = {
      sessionsStarted: 0,
      utterances: 0,
      interruptions: 0,
      llmCalls: 0,
      llmErrors: 0,
      deterministicAnswers: 0,
      groundedAnswers: 0
    };
    this.latencies = [];
  }

  inc(key) {
    if (typeof this.counters[key] !== "number") return;
    this.counters[key] += 1;
  }

  observeLatency(ms) {
    const value = Number(ms || 0);
    if (!value || value < 0) return;
    this.latencies.push(value);
    if (this.latencies.length > 500) this.latencies.shift();
  }

  latencySnapshot() {
    if (!this.latencies.length) return { p50: 0, p90: 0, p95: 0, count: 0 };
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const pick = (p) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
    return {
      p50: Math.round(pick(0.5)),
      p90: Math.round(pick(0.9)),
      p95: Math.round(pick(0.95)),
      count: sorted.length
    };
  }

  snapshot() {
    return {
      upMs: Date.now() - this.startedAt,
      counters: this.counters,
      latencyMs: this.latencySnapshot()
    };
  }
}

export const runtimeMetricsService = new RuntimeMetricsService();
