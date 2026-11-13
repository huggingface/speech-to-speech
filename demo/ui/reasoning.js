// @ts-check

export const REASONING_LEVELS = [
  ["default", "Server default"],
  ["none", "Off"],
  ["minimal", "Minimal"],
  ["low", "Low"],
  ["medium", "Medium"],
  ["high", "High"],
  ["xhigh", "Extra high"],
];
const STORAGE_KEY = "s2s.ws.reasoningEffort";

/** @param {string | null} effort */
function reasoningLabel(effort) {
  return REASONING_LEVELS.find(([value]) => value === (effort ?? "default"))?.[1] ?? "Server default";
}

/** @param {unknown} value @returns {string | null} */
export function normalizeReasoningEffort(value) {
  return REASONING_LEVELS.some(([effort]) => effort !== "default" && effort === value)
    ? /** @type {string} */ (value) : null;
}

export class ReasoningControl {
  /** @param {(effort: string | null) => void} onChange */
  constructor(onChange) {
    this.effort = normalizeReasoningEffort(localStorage.getItem(STORAGE_KEY));
    this._active = false;
    this._status = /** @type {HTMLElement} */ (document.querySelector("#reasoning-status"));
    const button = /** @type {HTMLButtonElement} */ (document.querySelector("#reasoning-btn"));
    const value = /** @type {HTMLElement} */ (document.querySelector("#reasoning-value"));
    const picker = /** @type {HTMLElement} */ (document.querySelector("#reasoning-picker"));
    const choices = /** @type {HTMLElement} */ (picker.querySelector(".reasoning-choices"));
    for (const [effort, label] of REASONING_LEVELS) {
      const row = document.createElement("label");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "reasoning-effort";
      radio.value = effort;
      row.append(radio, label);
      choices.append(row);
      radio.addEventListener("change", () => {
        this.effort = normalizeReasoningEffort(effort);
        if (this.effort === null) localStorage.removeItem(STORAGE_KEY);
        else localStorage.setItem(STORAGE_KEY, this.effort);
        render();
        this._awaitConfirmation();
        onChange(this.effort);
        picker.hidePopover();
        button.focus();
      });
    }
    const render = () => {
      value.textContent = reasoningLabel(this.effort);
      for (const radio of choices.querySelectorAll("input")) {
        radio.checked = radio.value === (this.effort ?? "default");
      }
    };
    picker.addEventListener("toggle", () => {
      const open = picker.matches(":popover-open");
      button.setAttribute("aria-expanded", String(open));
      if (open) /** @type {HTMLInputElement | null} */ (choices.querySelector(":checked"))?.focus();
    });
    render();
    this._awaitConfirmation();
  }

  /** @param {boolean} active */
  setActive(active) {
    this._active = active;
    this._awaitConfirmation();
  }

  _awaitConfirmation() {
    this._status.dataset.state = this._active ? "pending" : "idle";
    this._status.textContent = this._active ? "Waiting for server confirmation…" : "Applies when you start a conversation";
  }

  /** Accept only the server's session.updated echo, never the outgoing selection.
   * @param {unknown} reasoning */
  confirm(reasoning) {
    if (!this._active) return;
    const effort = normalizeReasoningEffort(/** @type {{effort?: unknown} | null} */ (reasoning)?.effort);
    const isDefault = reasoning === null || reasoning === undefined;
    if ((effort !== null && effort === this.effort) || (isDefault && this.effort === null)) {
      this._status.dataset.state = "confirmed";
      this._status.textContent = `✓ Server confirmed: ${reasoningLabel(this.effort)}`;
    } else {
      this._status.dataset.state = "unconfirmed";
      this._status.textContent = effort !== null
        ? `Server reports ${reasoningLabel(effort)} · waiting for ${reasoningLabel(this.effort)}`
        : "Server did not confirm this reasoning level";
    }
  }

  /** @param {string} message */
  onServerError(message) {
    if (!this._active || !/reasoning|thinking/i.test(message)) return;
    this._status.dataset.state = "unconfirmed";
    this._status.textContent = "Server reported a reasoning error";
  }
}
