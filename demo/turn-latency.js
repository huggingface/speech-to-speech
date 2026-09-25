// @ts-check
/**
 * @typedef {Object} TurnLatency
 * @property {number} version
 * @property {string} turn_id
 * @property {number} turn_revision
 * @property {string} response_key
 * @property {string} status
 * @property {number|null} stt_s
 * @property {number|null} llm_s
 * @property {number|null} tts_ttfa_s
 * @property {number|null} e2e_s
 * @property {number|null} mlx_lock_wait_s
 */

/** Read only terminal server measurements; older servers simply omit them.
 * @param {any} response
 * @returns {TurnLatency|null}
 */
export function readTurnLatency(response) {
  const raw = response?.metadata?.["speech_to_speech.turn_latency"];
  if (typeof raw !== "string") return null;
  try {
    const data = JSON.parse(raw);
    if (!data || data.version !== 1 ||
        typeof data.turn_id !== "string" || !data.turn_id ||
        typeof data.response_key !== "string" || !data.response_key ||
        !Number.isInteger(data.turn_revision) || data.turn_revision < 0 ||
        !["completed", "cancelled", "failed", "incomplete"].includes(data.status) ||
        data.status !== response.status) return null;
    for (const field of ["stt_s", "llm_s", "tts_ttfa_s", "e2e_s", "mlx_lock_wait_s"]) {
      if (data[field] !== null &&
          (typeof data[field] !== "number" || !Number.isFinite(data[field]) || data[field] < 0)) return null;
    }
    return data;
  } catch {
    return null;
  }
}
