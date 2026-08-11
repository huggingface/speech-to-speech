// @ts-check

/**
 * @typedef {Object} ToolExecutionResult
 * @property {string} callId
 * @property {string} output
 * @property {string} [image]
 */

/** Collect every tool execution from one response before requesting its follow-up. */
export class ToolCallBatcher {
  /** @param {(results: ToolExecutionResult[]) => void | Promise<void>} onReady */
  constructor(onReady) {
    this._onReady = onReady;
    /** @type {Map<string, { executions: Promise<ToolExecutionResult>[]; flush: Promise<void> | null }>} */
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
      batch = { executions: [], flush: null };
      this._batches.set(responseId, batch);
    }
    batch.executions.push(execution);
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
      for (const execution of batch.executions) void execution.catch(() => {});
      return null;
    }
    if (batch.flush) return batch.flush;

    batch.flush = Promise.all(batch.executions)
      .then((results) => this._onReady(results))
      .finally(() => {
        if (this._batches.get(responseId) === batch) this._batches.delete(responseId);
      });
    return batch.flush;
  }
}
