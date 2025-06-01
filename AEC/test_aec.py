"""
test_aec.py
---------------------------------
使用本地麦克风实时测试 **LivekitAecHandler** 的示例脚本。

> **更新 2025‑05‑31**
> • 彻底排空 `queue_out`，解决只生成约 1 s 音频的问题。
> • 代码整理：简化 flush/dump 逻辑。

步骤
~~~~
1. **采集** – 以 16 kHz / 16‑bit 单声道抓麦克风，10 ms 一块放入 `queue_in`。
2. **处理** – 后台线程不断 `handler.process()`，结果写入 `queue_out`。
3. **回放** – 主线程实时消费 `queue_out`，用 `sd.play()` 听消回声后音频。
4. **保存** – 录制结束后排空余下 `queue_out`，再连同 `handler.flush()` 的残帧写成 WAV。

依赖
----
```bash
pip install sounddevice numpy
```

用法
----
```bash
python test_aec.py               # 默认录 10 s
python test_aec.py --seconds 30  # 录 30 s
```
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Iterable, Union, Any

import numpy as np
import sounddevice as sd

# 请把包含 LivekitAecHandler 的路径加入 PYTHONPATH
from AEC.livekit_aec_handler import LivekitAecHandler

BytesLike = Union[bytes, bytearray]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _iter_flush_result(result: Any) -> Iterable[BytesLike]:
    """把 `handler.flush()` 的各种返回形态转换成 bytes 块序列。"""
    if result is None:
        return []

    if isinstance(result, (bytes, bytearray)):
        return [result]

    if isinstance(result, np.ndarray):
        return [result.astype(np.int16).tobytes()]

    if isinstance(result, (list, tuple)):
        out: list[BytesLike] = []
        for item in result:
            if isinstance(item, (bytes, bytearray)):
                out.append(item)
            elif isinstance(item, (int, np.integer)):
                out.append(np.array([item], dtype=np.int16).tobytes())
            elif isinstance(item, np.ndarray):
                out.append(item.astype(np.int16).tobytes())
        return out

    return []

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def test_livekit_aec(
    seconds: int = 10,
    sample_rate: int = 16_000,
    chunk_ms: int = 10,
    output_wav: str | Path = "aec_output.wav",
):
    stop_event = threading.Event()
    queue_in: queue.Queue[BytesLike] = queue.Queue()
    queue_out: queue.Queue[BytesLike] = queue.Queue()

    handler = LivekitAecHandler(stop_event, queue_in, queue_out)
    handler.setup()

    # 处理线程
    def worker() -> None:
        while not stop_event.is_set() or not queue_in.empty():
            try:
                raw = queue_in.get(timeout=0.05)
            except queue.Empty:
                continue
            for processed in handler.process(raw):
                queue_out.put(processed)

    threading.Thread(target=worker, daemon=True).start()

    chunk_samples = int(sample_rate * chunk_ms / 1000)
    processed_frames: list[np.ndarray] = []

    def audio_callback(indata, frames, *_):
        queue_in.put(indata.copy().tobytes())

    # ------------------- 录制/实时播放 ------------------- #
    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=chunk_samples,
        dtype="int16",
        callback=audio_callback,
    ):
        print(f"⏺️  正在录制并实时处理 {seconds}s …")
        end_time = time.time() + seconds
        while time.time() < end_time:
            try:
                block = queue_out.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_frames.append(np.frombuffer(block, dtype=np.int16))
            sd.play(processed_frames[-1], sample_rate, blocking=False)

    # ------------------- 收尾：排空队列 ------------------- #
    stop_event.set()

    # 等 worker 把残余 queue_in 消化完
    time.sleep(0.2)

    # 把 queue_out 里剩下的全部拿出来
    while not queue_out.empty():
        processed_frames.append(
            np.frombuffer(queue_out.get_nowait(), dtype=np.int16)
        )

    # flush() 里可能还有帧
    for leftover in _iter_flush_result(handler.flush()):
        processed_frames.append(np.frombuffer(leftover, dtype=np.int16))

    # ------------------- 写文件 ------------------- #
    if processed_frames:
        audio_out = np.concatenate(processed_frames)
        with wave.open(str(output_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_out.tobytes())
        print(f"✅ 完成！已保存 {len(audio_out)/sample_rate:.2f}s 到: {output_wav}")
    else:
        print("⚠️ 未捕获到任何处理数据")

    sd.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Livekit AEC handler with microphone input")
    parser.add_argument("--seconds", type=int, default=10, help="录制/处理时长（秒）")
    parser.add_argument("--output", type=str, default="aec_output.wav", help="输出文件名")
    args = parser.parse_args()

    test_livekit_aec(seconds=args.seconds, output_wav=args.output)