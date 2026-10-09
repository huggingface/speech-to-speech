// @ts-check
/**
 * Pure, stateless helpers for the WebSocket realtime client: transcript extraction from a
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
