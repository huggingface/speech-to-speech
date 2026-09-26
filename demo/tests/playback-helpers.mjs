import fs from "node:fs";
import vm from "node:vm";

export function worklet(onMessage = () => {}, sampleRate = 48_000) {
  let Processor;
  const reports = [];
  vm.runInNewContext(fs.readFileSync(new URL("../worklets/audio-playback.js", import.meta.url), "utf8"), {
    sampleRate,
    Float32Array,
    AudioWorkletProcessor: class {
      constructor() { this.port = { postMessage: (message) => { reports.push(message); onMessage(message); } }; }
    },
    registerProcessor(_name, value) { Processor = value; },
  });
  const processor = new Processor();
  return {
    reports,
    send(message) { processor.port.onmessage({ data: message }); },
    render(frames) {
      const result = [];
      while (frames > 0) {
        const block = new Float32Array(Math.min(128, frames));
        processor.process([], [[block]]);
        result.push(...block);
        frames -= block.length;
      }
      return result;
    },
  };
}

