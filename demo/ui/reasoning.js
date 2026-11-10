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

/** @param {unknown} value @returns {string | null} */
export function normalizeReasoningEffort(value) {
  return REASONING_LEVELS.some(([effort]) => effort !== "default" && effort === value)
    ? /** @type {string} */ (value) : null;
}

export class ReasoningControl {
  /** @param {(effort: string | null) => void} onChange */
  constructor(onChange) {
    this.effort = normalizeReasoningEffort(localStorage.getItem(STORAGE_KEY));
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
        onChange(this.effort);
        picker.hidePopover();
        button.focus();
      });
    }
    const render = () => {
      value.textContent = REASONING_LEVELS.find(([effort]) => effort === (this.effort ?? "default"))?.[1] ?? "Server default";
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
  }
}
