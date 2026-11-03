/** A per-connection gate: install before connecting so an early ready is retained. */
export function modelReadiness({ timeoutMs = 190_000 } = {}) {
  let settled = false;
  let resolveReady;
  let rejectReady;
  let timer;
  const promise = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  // Errors can arrive while the SDK is still establishing its session.
  void promise.catch(() => {});
  const finish = (error) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    if (error) rejectReady(error);
    else resolveReady();
  };
  timer = setTimeout(() => finish(new Error("Preparing models timed out. Please retry.")), timeoutMs);
  return {
    promise,
    event(event) {
      if (event?.type === "speech_to_speech.models.ready") finish();
      if (event?.type === "speech_to_speech.models.error") {
        finish(new Error(event.error?.message || event.message || "Could not prepare models. Please retry."));
      }
    },
    fail: finish,
    cancel() {
      finish(Object.assign(new Error("Model preparation cancelled"), { code: "aborted" }));
    },
  };
}
