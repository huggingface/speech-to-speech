import { spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import process from "node:process";
import readline from "node:readline";

export const SDK_CONFIG = {
  outputModalities: ["audio"],
  audio: {
    input: {
      format: { type: "audio/pcm", rate: 24_000 },
      transcription: { model: "whisper-1" },
      turnDetection: { type: "server_vad", interruptResponse: true },
      noiseReduction: null,
    },
    output: {
      format: { type: "audio/pcm", rate: 24_000 },
      voice: "coral",
      speed: 1,
    },
  },
};

export async function startTestServer() {
  const repository = path.resolve(import.meta.dirname, "..", "..");
  const python = process.env.S2S_PYTHON || path.join(repository, ".venv", "bin", "python");
  const child = spawn(
    python,
    ["-m", "tests.openai_realtime.agents_sdk_server"],
    { cwd: repository, stdio: ["ignore", "pipe", "pipe"] },
  );
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
    if (process.env.S2S_TEST_SERVER_LOGS === "1") process.stderr.write(chunk);
  });
  const lines = readline.createInterface({ input: child.stdout });
  const timer = setTimeout(() => child.kill("SIGTERM"), 60_000);
  try {
    const [line] = await Promise.race([
      once(lines, "line"),
      once(child, "exit").then(([code]) => {
        throw new Error(`test server exited before startup (${code}): ${stderr}`);
      }),
    ]);
    return {
      ...JSON.parse(line),
      async close() {
        clearTimeout(timer);
        child.kill("SIGTERM");
        await Promise.race([once(child, "exit"), new Promise((resolve) => setTimeout(resolve, 5_000))]);
      },
    };
  } catch (error) {
    clearTimeout(timer);
    child.kill("SIGTERM");
    throw error;
  }
}

export async function waitFor(predicate, { timeout = 10_000, interval = 25 } = {}) {
  const deadline = Date.now() + timeout;
  let value;
  while (Date.now() < deadline) {
    value = await predicate();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
  throw new Error(`condition not met within ${timeout} ms; last value: ${JSON.stringify(value)}`);
}

export async function post(baseUrl, pathName) {
  const response = await fetch(`${baseUrl}${pathName}`, { method: "POST" });
  if (!response.ok) throw new Error(`${pathName} failed: ${response.status} ${await response.text()}`);
  return response.json();
}

export async function state(baseUrl) {
  const response = await fetch(`${baseUrl}/test/state`);
  if (!response.ok) throw new Error(`/test/state failed: ${response.status}`);
  return response.json();
}
