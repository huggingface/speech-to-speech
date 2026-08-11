// @ts-check

/**
 * @typedef {Object} ToolExecutionResult
 * @property {string} callId
 * @property {string} output
 * @property {string} [image]
 */

/** Coordinate one queued follow-up and ordered tool results per response. */
export class ToolCallBatcher {
  /**
   * @param {(responseId: string, result: ToolExecutionResult) => void | Promise<void>} onResult
   * @param {(responseId: string) => void} onFollowUp
   */
  constructor(onResult, onFollowUp) {
    this._onResult = onResult;
    this._onFollowUp = onFollowUp;
    /** @type {Map<string, { delivery: Promise<void>; flush: Promise<void> | null }>} */
    this._batches = new Map();
  }

  /**
   * Register a tool execution in the order its call appeared in the response.
   * @param {string} responseId
   * @param {Promise<ToolExecutionResult>} execution
   */
  add(responseId, execution) {
    let batch = this._batches.get(responseId);
    if (!batch) {
      batch = { delivery: Promise.resolve(), flush: null };
      this._batches.set(responseId, batch);
      this._onFollowUp(responseId);
    }
    // Preserve function-call order even when executions finish out of order.
    // The server releases the queued response only after every call is paired,
    // so later calls can still extend this chain safely.
    batch.delivery = batch.delivery
      .then(() => execution)
      .then(async (result) => {
        if (this._batches.get(responseId) !== batch) return;
        await this._onResult(responseId, result);
      });
  }

  /**
   * Finish the originating response. Results are already delivered as soon as
   * they settle; this terminal only cleans up or suppresses late cancelled work.
   * @param {string} responseId
   * @param {string} status
   * @returns {Promise<void> | null}
   */
  finish(responseId, status) {
    const batch = this._batches.get(responseId);
    if (!batch) return null;
    if (status !== "completed") {
      this._batches.delete(responseId);
      // Executions cannot be cancelled, but the identity check above suppresses
      // their delivery and this catch prevents a discarded rejection leaking.
      void batch.delivery.catch(() => {});
      return null;
    }
    if (batch.flush) return batch.flush;

    batch.flush = batch.delivery
      .finally(() => {
        if (this._batches.get(responseId) === batch) this._batches.delete(responseId);
      });
    return batch.flush;
  }

  /** Suppress every outstanding result after barge-in starts a newer turn. */
  discardAll() {
    for (const [responseId, batch] of this._batches) {
      this._batches.delete(responseId);
      void batch.delivery.catch(() => {});
    }
  }
}
