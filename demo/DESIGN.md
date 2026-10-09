# Design language

The reference for keeping this app visually coherent as it grows. Read it before
touching `style.css`, `index.html`, or any DOM-building code in `main.js`. Every
rule here is already live in the codebase — this file explains the *why* so changes
extend the system instead of drifting from it.

---

## The thesis: color belongs to the voice

This is a voice app. The one thing in the room that should have color is the thing
that is talking. So:

- **The orb** carries saturated color. It glows, and the glow's hue changes with
  conversational state.
- **Everything else is monochrome** — a precise cool-grey dark canvas. Surfaces,
  borders, buttons, panels, the transcript: all greyscale.
- The **only** exceptions are tiny *role echoes* (a one-word mono label, a small
  icon) that borrow the orb's state hue so the transcript reads in the same color
  language the orb speaks. They are accents the size of a word, never fills.
- **Brand logos keep their own color.** The Hugging Face mark (`#ffd21e`) and the
  Cerebras mark (`#f15a29`) render in their brand colors in the identity credits and
  the about panel — a deliberate, owner-approved exception. It applies to those two
  logos only; do not generalize it to other chrome.

If you find yourself adding a tinted background, a colored border, or a bright
button anywhere outside the orb, stop — that color almost certainly belongs to the
orb instead, or shouldn't exist.

---

## Color tokens

All defined in `:root` in `style.css`. Use the variables, never raw hex in rules.

### Canvas (the monochrome world)
| Token | Value | Use |
|---|---|---|
| `--bg` | `#0a0b10` | Page background (a cool near-black) |
| `--bg-elev` | `#13151c` | Raised surfaces: bubbles, panels, icon buttons |
| `--bg-elev-2` | `#1b1e29` | Surfaces on surfaces: history bodies, inputs |
| `--border` | `rgba(255,255,255,.08)` | Default hairline |
| `--border-strong` | `rgba(255,255,255,.16)` | Emphasised hairline |
| `--text` | `#f5f6fa` | Primary text; also the *primary button* fill |
| `--text-dim` | `rgba(245,246,250,.65)` | Secondary text |
| `--text-faint` | `rgba(245,246,250,.42)` | Captions, labels, footer |

### Voice (the only saturated hues)
These are the orb's state colors. They appear on the orb, and as small role echoes
in the transcript — nowhere else.

| Token | Value | Meaning |
|---|---|---|
| `--accent` / `--speaking` | `#8b7dff` violet | Assistant speaking |
| `--accent-2` / `--listening` | `#22d3ee` cyan | You / listening |
| `--processing` | `#f59e0b` amber | Thinking / tool call |
| `--error` | `#ff6a75` | Error |
| `--success` | `#34d399` | Ready / connected |

### Role echoes (semantic aliases — use these in chat code)
| Token | Maps to | Where it shows |
|---|---|---|
| `--voice-user` | cyan | `YOU` label + user bubble accents |
| `--voice-assistant` | violet | `ASSISTANT` label + assistant accents |
| `--voice-tool` | amber | `TOOL CALL` label, wrench icon |

**Why this mapping:** it mirrors the orb exactly — when *you* speak the orb is cyan
(`state-listening`), when the *assistant* speaks it's violet (`state-ai-speaking`),
when it's working it's amber (`state-processing`). The transcript is a quiet replay
of the orb's color story.

### Orb state → glow (`.circle.state-*` → `--glow`)
| State | Glow |
|---|---|
| `signed-out` | violet `#8b7dff` |
| `authenticated` / `ready` | green `#34d399` |
| `connecting` / `connected` / `starting` | yellow `#facc15` |
| `listening` / `user-speaking` | cyan (`--listening`) |
| `processing` | amber (`--processing`) |
| `ai-speaking` | violet (`--speaking`) |
| `error` | red (`--error`) |

Adding a new state? Give it a `--glow`, and if it surfaces in the transcript, add a
matching `--voice-*` alias rather than a one-off color.

---

## Typography

Two faces, two jobs. Never reach for a third.

- **Inter** — body and UI. Wordmark, buttons, inputs, panel titles, prose, history
  message bodies. The workhorse; it should feel neutral and get out of the way.
- **Geist Mono** (`--font-mono`) — the **machine voice**. Reserved for text the
  *system* emits or identifiers it reports, never for human prose.

### When mono is correct
Mono signals "this is the machine talking or naming itself." Use it for:
- the orb's status caption (`.circle-caption`)
- role eyebrows (`YOU` / `ASSISTANT` / `TOOL CALL`)
- tool-call names and argument JSON
- the `·WebSocket` transport tag, bitrate readouts, connection identifiers
- the empty-state label

Mono text is set uppercase with `letter-spacing: ~0.1–0.14em` and weight `500`, so it
reads as a typed status line, not a headline. Body copy, button labels, and
explanatory `small` text stay **Inter** — putting prose in mono breaks the metaphor.

The font is loaded in `index.html`; the stack falls back to system mono gracefully if
the CDN is blocked.

---

## Layout

- **One continuous canvas.** No dividers under the topbar or above the footer, no
  panel chrome competing with content. The topbar and footer float over the stage.
  Keep it that way — a new section earns a hairline (`--border`) only if it genuinely
  needs separating.
- **The orb is the hero and the center of gravity.** It sits dead-center on the
  stage. Controls flank it (mic / stop), captions sit beneath. Don't crowd it.
- **Hairlines, not boxes.** Separation comes from `1px` borders at 8–16% white and
  from spacing, not from heavy fills or shadows. Shadows are soft and low
  (`0 4px 18px rgba(0,0,0,.32)`), used only to lift floating elements (bubbles,
  panels, modal).
- **Radii:** `--radius-sm: 8px` (buttons, inputs, chips), `--radius-md: 14px`
  (bubbles, message bodies, modal), `--radius-lg: 22px` (reserved). Pick by element
  size; don't invent new values.
- **Two reading surfaces for the transcript:** ephemeral bubbles top-right (desktop
  only) that log and fade, and a slide-in history panel for review. On phones the
  bubble stream is dropped and the panel goes full-screen — the panel is the single
  source of truth there.

---

## Components

- **Buttons.** Default (`.btn`) is a neutral elevated surface. The *primary* button
  is **near-white on dark** (`--text` fill, `--bg` text) — the highest-contrast thing
  on the page that *isn't* the orb. There is no colored button; emphasis comes from
  contrast, not hue.
- **Icon buttons** (`.icon-btn`) are `36px`, elevated surface, dim icon that brightens
  on hover. Side controls (`.side-btn`) are circular, collapse to zero size until the
  session is live (by width on desktop, by height in the mobile column).
- **Chat bubbles & history messages** share one neutral surface. They are
  distinguished by **side** (you = left, assistant = right) plus the **mono role
  label** in the role-echo hue — not by tinted fills. Tool entries use the wrench icon
  + mono + amber, on the same neutral surface.
- **Badge** (new-message dot) is monochrome white — a signal, not a color accent.
- **Focus** is visible and neutral: inputs focus to `--text-dim`; the orb uses a
  `--glow`-colored outline (it's the orb, so color is allowed).

---

## Motion

- **The orb is audio-reactive, not timer-driven.** Mic RMS (`--audio-level`) and the
  assistant output level (`--ai-audio-level`) drive scale/opacity at display rate, so
  every syllable moves it. This is the signature animation — keep new motion
  subordinate to it.
- **Quiet by default.** Breathing/glow throbs are slow (1.4–2.4s) and low-contrast.
  Resist adding scattered micro-animations; an orchestrated moment beats many small
  ones, and excess motion reads as AI-generated.
- **First paint is frozen.** `body.booting` disables all transitions until the first
  frame commits (stripped after one rAF in `main.js`). Anything new that would
  otherwise animate-in on load must respect this.
- Honor `prefers-reduced-motion` for any motion you add.

---

## Writing / copy voice

- Sentence case, plain verbs, no filler. Tuned and quiet — match the minimal canvas.
- Name things by what the user controls, not by the system's internals. A button says
  exactly what it does, and keeps the same word through the flow.
- **Empty states invite action** ("Tap the orb and start talking"), they don't just
  set a mood.
- **Errors state what happened and how to recover**, in the interface's voice — they
  don't apologize and are never vague.
- Mono labels are terse identifiers (`YOU`, `TOOL CALL`); prose stays in Inter.

---

## Responsive floor (non-negotiable)

Every change ships meeting these:
- Works down to a `360px`-wide phone. The `@media (max-width: 600px)` block already
  handles the phone layout — extend it, don't fight it.
- Visible keyboard focus on every interactive element.
- `prefers-reduced-motion` respected.
- Tap targets ≥ `44px`; `touch-action: manipulation` on anything tappable.

---

## Before you ship — the mirror check

1. Is every saturated color either on the orb or a word-sized role echo? If a fill or
   border is colored, remove that accessory.
2. Is mono used only for machine/system text, and Inter for everything human?
3. Does any new state have both a `--glow` and (if it appears in chat) a `--voice-*`
   alias?
4. Are separations hairlines + spacing, not boxes and heavy shadows?
5. Did you add motion? Is it quieter than the orb and reduced-motion-safe?
6. Remove one accessory. The minimal look survives on precision, not addition.
