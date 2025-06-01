import argparse
import queue
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

# 将 AEC 处理器所在路径加入 PYTHONPATH，或把项目根目录放到环境变量里
from AEC.livekit_aec_handler import LivekitAecHandler


def test_livekit_aec(
    seconds: int = 10,
    sample_rate: int = 16_000,
    chunk_ms: int = 10,
    output_wav: str | Path = "aec_output.wav",
):
    """
    使用 LivekitAecHandler 对麦克风输入做 AEC 处理，并保存结果。

    Args:
        seconds: 采集/处理时长（秒）
        sample_rate: 采样率，需与 AEC handler 保持一致 (16 kHz)
        chunk_ms: 送入 AEC 的块大小，单位 ms。LivekitAecHandler 需 10 ms（=320 B）
        output_wav: 输出 WAV 文件路径
    """
    handler = LivekitAecHandler()
    handler.setup()

    # 计算每块的采样点数
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    # sounddevice 既支持 int16 ndarray，也可 bytes；我们用 int16 方便后续拼接
    in_queue: queue.Queue[np.ndarray] = queue.Queue()
    processed_frames: list[np.ndarray] = []

    def audio_callback(indata, frames, time_info, status):
        """
        输入流回调函数，将麦克风块放进队列。
        """
        if status:
            print(f"[StreamWarning] {status}", flush=True)
        # 这里的 indata shape: (frames, channels)
        in_queue.put(indata.copy())

    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=chunk_samples,
        dtype="int16",
        callback=audio_callback,
    ):
        print(f"⏺️  正在录制并实时处理 {seconds}s ...")
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                block = in_queue.get(timeout=0.1)  # ndarray(int16)
            except queue.Empty:
                continue

            # ndarray → bytes
            raw_bytes = block.tobytes()

            for processed_block in handler.process(raw_bytes):
                processed_frames.append(
                    np.frombuffer(processed_block, dtype=np.int16)
                )
                # 实时回放（可注释）
                sd.play(
                    processed_frames[-1],
                    sample_rate,
                    blocking=False,
                )

        # flush 未完成的残余数据
        leftover = handler.flush()
        if leftover:
            processed_frames.append(np.frombuffer(leftover, dtype=np.int16))

    # 拼接所有块
    if processed_frames:
        audio_out = np.concatenate(processed_frames)
        # 保存 WAV
        with wave.open(str(output_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_out.tobytes())
        print(f"✅ 完成！输出文件已保存: {output_wav}")
    else:
        print("⚠️ 未捕获到任何处理数据")

    # 确保所有播放停止
    sd.stop()

    return Path(output_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Livekit AEC handler with microphone input")
    parser.add_argument("--seconds", type=int, default=10, help="录制/处理时长，单位秒")
    parser.add_argument("--output", type=str, default="aec_output.wav", help="输出文件名")
    args = parser.parse_args()