// @ts-check

/**
 * @typedef {Object} ToolExecutionResult
 * @property {string} callId
 * @property {string} output
 * @property {string} [image]
 */

/** Deliver tool results early, then request one follow-up after response.done. */
export class ToolCallBatcher {
  /**
   * @param {(result: ToolExecutionResult) => void | Promise<void>} onResult
   * @param {() => void | Promise<void>} onReady
   */
  constructor(onResult, onReady) {
    this._onResult = onResult;
    this._onReady = onReady;
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
    }
    // Keep protocol items in function-call order even when tools finish out of
    // order. Each result is still sent as soon as every earlier result is ready.
    batch.delivery = batch.delivery
      .then(() => execution)
      .then(async (result) => {
        if (this._batches.get(responseId) !== batch) return;
        await this._onResult(result);
      });
  }

  /**
   * Finish the originating response. Completed responses flush once all tools
   * settle; unsuccessful responses discard calls the backend rolled back.
   * @param {string} responseId
   * @param {string} status
   * @returns {Promise<void> | null}
   */
  finish(responseId, status) {
    const batch = this._batches.get(responseId);
    if (!batch) return null;
    if (status !== "completed") {
      this._batches.delete(responseId);
      // Executions cannot be cancelled, but a discarded rejection should not
      // become unhandled after the response is gone.
      void batch.delivery.catch(() => {});
      return null;
    }
    if (batch.flush) return batch.flush;

    batch.flush = batch.delivery
      .then(() => this._onReady())
      .finally(() => {
        if (this._batches.get(responseId) === batch) this._batches.delete(responseId);
      });
    return batch.flush;
  }
}
