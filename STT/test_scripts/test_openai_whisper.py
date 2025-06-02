"""
用 handler.run() 方式实时测试 OpenAIRealtimeSTTHandler
------------------------------------------------------

• 录音: sounddevice 16 kHz / 20 ms
• 推送: 麦克风帧 → queue_in
• 处理: OpenAIRealtimeSTTHandler.run() 在线程内自动执行 process()
• 输出: 主线程从 queue_out 取 (text, lang) 并打印
"""

from __future__ import annotations

import argparse
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from STT.openai_whisper_handler import OpenAITTSHandler

SAMPLE_RATE = 16_000
CHUNK_MS = 1000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


def main():
    # ---------- 队列 & 事件 ---------- #
    stop_evt = threading.Event()
    q_in: queue.Queue[np.ndarray | bytes] = queue.Queue()
    q_out: queue.Queue[tuple[str, str] | bytes] = queue.Queue()

    # ---------- Handler 实例 ---------- #
    handler = OpenAITTSHandler(
        stop_evt,
        q_in,
        q_out,
        setup_kwargs=dict(
            model="gpt-4o-mini-transcribe",
            language=None
        )
    )

    # thread.run() 会在内部 while 循环里不断 get → process → put
    th = threading.Thread(target=handler.run, daemon=True)
    th.start()

    # ---------- 麦克风采集 ---------- #
    def mic_cb(indata, frames, *_):
        q_in.put(indata.copy().squeeze())     # float32 ndarray

    print(f"🎙  Speak (auto-stop {5}s)…")
    last_print = time.time()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=mic_cb,
        ):
            end = time.time() + 5
            while time.time() < end:
                try:
                    item = q_out.get(timeout=0.1)
                except queue.Empty:
                    continue

                # handler.run() 会在结束时放 b\"END\"
                if isinstance(item, bytes) and item == b"END":
                    break

                text, lang = item
                print(f"[{lang}] {text}")
                last_print = time.time()
    except KeyboardInterrupt:
        print("✋ user stop.")
    finally:
        # -------- 收尾 -------- #
        stop_evt.set()
        q_in.put(b"END")     # 让 handler.run() 跳出 while
        th.join(timeout=2)
        print("✅ handler stopped.")


if __name__ == "__main__":
    main()
