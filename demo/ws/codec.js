// @ts-check
/**
 * Pure, stateless helpers for the WebSocket realtime client: base64 <-> PCM
 * conversion for the audio frames on the wire, transcript extraction from a
 * `response.done` payload, and a tiny URL helper. Kept separate from the client
 * so the protocol/state logic stays readable.
 */

/** @param {string} url */
export function trimTrailingSlash(url) {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

/**
 * Pull the assistant transcript out of a `response.done` payload. The text
 * lives in `response.output[].content[].transcript` (audio) or `.text`. Used as
 * the source of truth for interrupted replies, where the dedicated
 * `*.transcript.done` event may never arrive.
 * @param {any} response
 * @returns {string}
 */
export function extractResponseTranscript(response) {
  const output = response?.output;
  if (!Array.isArray(output)) return "";
  /** @type {string[]} */
  const parts = [];
  for (const item of output) {
    for (const part of item?.content ?? []) {
      const text = part?.transcript ?? part?.text;
      if (typeof text === "string" && text.trim()) parts.push(text.trim());
    }
  }
  return parts.join(" ").trim();
}

/** @param {ArrayBuffer} buf */
export function base64FromArrayBuffer(buf) {
  const bytes = new Uint8Array(buf);
  // Chunked encoding so we don't blow up the call stack on long buffers.
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, /** @type {number[]} */ (
      /** @type {unknown} */ (bytes.subarray(i, i + chunk))
    ));
  }
  return btoa(binary);
}

/** @param {string} b64 */
export function base64ToBytes(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = binary.charCodeAt(i);
  return out;
}
