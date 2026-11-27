// @ts-check
/** Tiny canvas confetti burst for streaks and good results. */

const COLORS = ["#1d4ed8", "#0d7a53", "#e07a16", "#c4283e", "#38bdf8", "#f59e0b"];

/** @param {{ count?: number }} [opts] */
export function confetti(opts = {}) {
  const canvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById("confetti"));
  if (!canvas || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = innerWidth * dpr;
  canvas.height = innerHeight * dpr;
  ctx.scale(dpr, dpr);
  const stage = document.getElementById("stage")?.getBoundingClientRect();
  const cx = stage ? stage.left + stage.width / 2 : innerWidth / 2;
  const cy = stage ? stage.top + stage.height * 0.35 : innerHeight / 3;
  const parts = Array.from({ length: opts.count ?? 100 }, () => {
    const a = Math.random() * Math.PI * 2;
    const v = 4 + Math.random() * 9;
    return {
      x: cx, y: cy,
      vx: Math.cos(a) * v, vy: Math.sin(a) * v - 6,
      r: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.4,
      w: 6 + Math.random() * 6, h: 8 + Math.random() * 8,
      c: COLORS[(Math.random() * COLORS.length) | 0],
    };
  });
  const t0 = performance.now();
  const frame = (/** @type {number} */ now) => {
    const t = now - t0;
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    for (const p of parts) {
      p.vy += 0.32;
      p.vx *= 0.985;
      p.x += p.vx;
      p.y += p.vy;
      p.r += p.vr;
      ctx.save();
      ctx.globalAlpha = Math.max(0, 1 - t / 2600);
      ctx.translate(p.x, p.y);
      ctx.rotate(p.r);
      ctx.fillStyle = p.c;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h * Math.abs(Math.cos(p.r * 2)));
      ctx.restore();
    }
    if (t < 2600) requestAnimationFrame(frame);
    else ctx.clearRect(0, 0, innerWidth, innerHeight);
  };
  requestAnimationFrame(frame);
}
