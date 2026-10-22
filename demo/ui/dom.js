// @ts-check
/** Small shared helpers used across the UI modules: a strict query selector,
 *  HTML escaping for text we drop into innerHTML, error-string trimming, and
 *  the opt-in debug flag. */

/** Opt-in tracing: `localStorage.setItem("s2s.debug", "1")` then reload. */
export const DEBUG = (() => {
  try {
    return localStorage.getItem("s2s.debug") === "1";
  } catch {
    return false;
  }
})();

/**
 * Query a single element, throwing if it's missing (so a broken selector fails
 * loudly at startup rather than as a later null-deref).
 * @template {HTMLElement} T
 * @param {string} selector
 * @returns {T}
 */
export function $(selector) {
  const el = document.querySelector(selector);
  if (!el) throw new Error(`Missing element: ${selector}`);
  return /** @type {T} */ (el);
}

/** @param {string} s @returns {string} */
export function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** User-facing error text, including SDK wrappers and opaque browser events.
 * @param {unknown} error
 * @returns {string}
 */
export function errorMessage(error) {
  let value = error;
  const seen = new Set();
  for (let depth = 0; depth < 5 && value != null; depth += 1) {
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value !== "object" || seen.has(value)) break;
    seen.add(value);
    const wrapped = /** @type {{message?: unknown, error?: unknown, detail?: {error?: unknown}}} */ (value);
    if (typeof wrapped.message === "string" && wrapped.message.trim()) return wrapped.message;
    value = wrapped.error ?? wrapped.detail?.error;
  }
  return "Voice connection failed. Reload the Space to refresh access and try again.";
}

/** Trim a long error message to fit the orb caption. @param {string} text */
export function truncateError(text) {
  if (text.length <= 90) return text;
  return text.slice(0, 87) + "…";
}
