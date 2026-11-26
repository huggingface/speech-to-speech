// JSZip ships no ES module build. The splat renderer imports it as a module,
// so load the UMD build once and hand back its global.
const ready = globalThis.JSZip
  ? Promise.resolve()
  : new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "vendor/jszip/jszip.min.js";
      s.onload = resolve;
      s.onerror = () => reject(new Error("JSZip failed to load"));
      document.head.append(s);
    });
await ready;
export default globalThis.JSZip;
