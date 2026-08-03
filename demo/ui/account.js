// @ts-check
/**
 * Account — the HF login chip and the daily-limit modal.
 *
 * Reads `/api/me` to learn the current tier (anonymous / signed-in / PRO) and
 * remaining daily talk-time, renders a sign-in pill or a signed-in chip with a
 * small popover (tier, remaining, sign out, upgrade), and shows the limit modal
 * when a conversation is refused or cut. The time metering itself lives in
 * main.js (heartbeat loop) + the server; this module is just the surface.
 *
 * Inert unless the deploy is in LB mode (`/api/me` → `{enabled:true}`).
 */

import { $, escHtml } from "./dom.js";

const PRO_URL = "https://huggingface.co/subscribe/pro";

// Official multi-color Hugging Face logo, used in the badge + sign-in CTA.
const HF_MARK = `<svg class="hf-logo" viewBox="0 0 95 88" fill="none" aria-hidden="true"><path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z"/><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"/><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"/><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"/><path fill="#3A3B45" fill-rule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clip-rule="evenodd"/><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"/><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"/><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"/><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"/></svg>`;

/** @param {number} sec @returns {string} "m:ss" */
function fmt(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export class Account {
  constructor() {
    /** @type {HTMLElement} */
    this._root = $("#account");
    /** @type {HTMLDialogElement} */
    this._modal = $("#limit-modal");
    this._modalTitle = $("#limit-title");
    this._modalMsg = $("#limit-msg");
    this._modalNote = $("#limit-note");
    /** @type {HTMLAnchorElement} */
    this._modalCta = /** @type {any} */ ($("#limit-cta"));

    /** @type {{enabled:boolean, auth?:boolean, loggedIn?:boolean, username?:string, avatar?:string, tier?:string, remainingSec?:number|null, limitSec?:number|null, loginUrl?:string|null, logoutUrl?:string|null}} */
    this._me = { enabled: false };
    this._popoverOpen = false;

    $("#limit-close").addEventListener("click", () => this._modal.close());
    this._modal.addEventListener("click", (e) => {
      if (e.target === this._modal) this._modal.close();
    });
    // Close the popover on an outside click.
    document.addEventListener("click", (e) => {
      if (this._popoverOpen && !this._root.contains(/** @type {Node} */ (e.target))) {
        this._closePopover();
      }
    });
  }

  get tier() {
    return this._me.tier || "anon";
  }

  /** Fetch `/api/me` and (re)render the chip. Safe to call repeatedly (load,
   *  after the OAuth redirect, after a conversation ends). */
  async refresh() {
    try {
      const res = await fetch("api/me");
      this._me = res.ok ? await res.json() : { enabled: false };
    } catch {
      this._me = { enabled: false };
    }
    this._render();
  }

  _render() {
    const me = this._me;
    if (!me.enabled) {
      this._root.hidden = true;
      this._root.innerHTML = "";
      return;
    }
    this._root.hidden = false;

    if (!me.loggedIn) {
      // Signed-out: a sign-in pill (only when OAuth is actually available).
      if (me.auth && me.loginUrl) {
        this._root.innerHTML = `<a class="signin-pill" href="${escHtml(me.loginUrl)}" title="Sign in for more time"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg><span>Sign in</span></a>`;
      } else {
        this._root.innerHTML = "";
        this._root.hidden = true;
      }
      return;
    }

    // Signed-in: avatar + handle chip that toggles a popover.
    const isPro = me.tier === "pro";
    // Org members get unlimited usage too, but aren't PRO — don't brand them so.
    const isUnlimited = isPro || me.tier === "org";
    const avatar = me.avatar
      ? `<img class="account-avatar" src="${escHtml(me.avatar)}" alt="" />`
      : `<span class="account-avatar account-avatar-fallback">${escHtml((me.username || "?")[0].toUpperCase())}</span>`;
    const remaining =
      isUnlimited || me.remainingSec == null
        ? "Unlimited"
        : `${fmt(me.remainingSec)} left today`;
    const tierLabel = isPro ? "PRO" : isUnlimited ? "Team" : "Free";

    this._root.innerHTML = `
      <button id="account-chip" class="account-chip" aria-haspopup="true" aria-expanded="false">
        ${avatar}
        <span class="account-handle">${escHtml(me.username || "you")}</span>
        ${isPro
          ? '<span class="account-pro">PRO</span>'
          : me.tier === "org"
            ? '<span class="account-pro account-team">TEAM</span>'
            : ""}
      </button>
      <div id="account-pop" class="account-pop" hidden>
        <div class="account-pop-row account-pop-name">${escHtml(me.username || "you")}</div>
        <div class="account-pop-row account-pop-meta">
          <span class="account-tier">${tierLabel}</span>
          <span class="account-remaining">${escHtml(remaining)}</span>
        </div>
        ${isUnlimited ? "" : `<a class="account-pop-link" href="${PRO_URL}" target="_blank" rel="noopener">Upgrade to PRO</a>`}
        <a class="account-pop-link account-signout" href="${escHtml(me.logoutUrl || "#")}">Sign out</a>
      </div>`;

    const chip = $("#account-chip");
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      this._popoverOpen ? this._closePopover() : this._openPopover();
    });
  }

  _openPopover() {
    const pop = document.getElementById("account-pop");
    const chip = document.getElementById("account-chip");
    if (!pop || !chip) return;
    pop.hidden = false;
    chip.setAttribute("aria-expanded", "true");
    this._popoverOpen = true;
  }

  _closePopover() {
    const pop = document.getElementById("account-pop");
    const chip = document.getElementById("account-chip");
    if (pop) pop.hidden = true;
    if (chip) chip.setAttribute("aria-expanded", "false");
    this._popoverOpen = false;
  }

  /**
   * Show the limit modal for a tier — used both when a conversation is refused
   * at start (402) and when a live one is cut (heartbeat `expired`).
   * @param {string} [tier]
   */
  showLimit(tier = this.tier) {
    const canSignIn = this._me.auth && this._me.loginUrl;
    this._modalTitle.textContent = "Thanks for chatting!";
    if (tier === "anon") {
      this._modalMsg.textContent =
        "Guest conversations run for 5 minutes. Sign in with Hugging Face to get 10 minutes a day for free, and PRO members chat with no limit at all.";
      this._modalNote.textContent = "Your free minutes refresh tomorrow.";
      if (canSignIn) {
        this._modalCta.innerHTML = `${HF_MARK}<span>Sign in with Hugging Face</span>`;
        this._modalCta.href = /** @type {string} */ (this._me.loginUrl);
        this._modalCta.hidden = false;
      } else {
        this._modalCta.hidden = true;
      }
    } else {
      // Signed-in, non-PRO.
      this._modalMsg.textContent =
        "You've enjoyed your 10 minutes for today. Go PRO for unlimited conversations and to support open source AI.";
      this._modalNote.textContent = "Or come back tomorrow. Your minutes reset daily.";
      this._modalCta.innerHTML = "<span>Upgrade to PRO</span>";
      this._modalCta.href = PRO_URL;
      this._modalCta.hidden = false;
    }
    if (!this._modal.open) this._modal.showModal();
  }

  /** Show a warm "we're at capacity" message when even the waiting line is full.
   *  Reuses the limit modal shell; no call-to-action, just reassurance. */
  showBusy() {
    this._modalTitle.textContent = "Hugged to the limit 🤗";
    this._modalMsg.textContent =
      "Every slot and the whole line are full right now. Too much love! Grab a coffee and pop back in a minute.";
    this._modalNote.textContent = "A spot usually opens up within a few minutes.";
    this._modalCta.hidden = true;
    if (!this._modal.open) this._modal.showModal();
  }
}
