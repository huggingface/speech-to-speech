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
    /** @type {Map<string, {
     *   calls: Map<string, { outputIndex: number; execution: Promise<ToolExecutionResult> | null; delivered: boolean }>;
     *   delivery: Promise<void>;
     *   flush: Promise<void> | null;
     *   hasItemOrdering: boolean;
     * }>} */
    this._batches = new Map();
  }

  /**
   * Register the protocol position announced by response.output_item.added.
   * @param {string} responseId
   * @param {string} callId
   * @param {number} outputIndex
   */
  register(responseId, callId, outputIndex) {
    const batch = this._batch(responseId);
    this._registerCall(batch, callId, outputIndex);
    batch.hasItemOrdering = true;
    this._pump(responseId, batch);
  }

  /**
   * Register a tool execution. Execution starts before this method is called;
   * result delivery follows the output-item order registered above.
   * @param {string} responseId
   * @param {string} callId
   * @param {number} outputIndex
   * @param {Promise<ToolExecutionResult>} execution
   */
  add(responseId, callId, outputIndex, execution) {
    const batch = this._batch(responseId);
    const call = this._registerCall(batch, callId, outputIndex);
    // A later call can reject while delivery is still waiting for an earlier
    // result. Observe it now without replacing the original promise, so the
    // ordered delivery chain still receives and surfaces the rejection.
    void execution.catch(() => {});
    if (call.execution) return;
    call.execution = execution;
    if (batch.hasItemOrdering) this._pump(responseId, batch);
  }

  /** @param {string} responseId */
  _batch(responseId) {
    let batch = this._batches.get(responseId);
    if (!batch) {
      batch = {
        calls: new Map(),
        delivery: Promise.resolve(),
        flush: null,
        hasItemOrdering: false,
      };
      this._batches.set(responseId, batch);
    }
    return batch;
  }

  /**
   * @param {{ calls: Map<string, { outputIndex: number; execution: Promise<ToolExecutionResult> | null; delivered: boolean }> }} batch
   * @param {string} callId
   * @param {number} outputIndex
   */
  _registerCall(batch, callId, outputIndex) {
    let call = batch.calls.get(callId);
    if (!call) {
      call = { outputIndex, execution: null, delivered: false };
      batch.calls.set(callId, call);
    } else {
      call.outputIndex = outputIndex;
    }
    return call;
  }

  /**
   * @param {string} responseId
   * @param {{
   *   calls: Map<string, { outputIndex: number; execution: Promise<ToolExecutionResult> | null; delivered: boolean }>;
   *   delivery: Promise<void>;
   * }} batch
   */
  _pump(responseId, batch) {
    batch.delivery = batch.delivery.then(async () => {
      while (this._batches.get(responseId) === batch) {
        const next = [...batch.calls.values()]
          .filter((call) => !call.delivered)
          .sort((left, right) => left.outputIndex - right.outputIndex)[0];
        if (!next?.execution) return;
        const result = await next.execution;
        if (this._batches.get(responseId) !== batch) return;
        await this._onResult(result);
        next.delivered = true;
      }
    });
    // finish() exposes this rejection to its caller, but the response may not
    // be terminal yet when a tool fails. Mark the current chain as observed in
    // the meantime without changing its eventual rejected state.
    void batch.delivery.catch(() => {});
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

    // Servers without response.output_item.added cannot release results early,
    // but response.done guarantees every arguments.done event has arrived, so
    // output_index is authoritative at this point.
    batch.hasItemOrdering = true;
    this._pump(responseId, batch);
    batch.flush = batch.delivery
      .then(() => this._onReady())
      .finally(() => {
        if (this._batches.get(responseId) === batch) this._batches.delete(responseId);
      });
    return batch.flush;
  }
}
