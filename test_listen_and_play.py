# realtime_aec_mic_test.py
# -------------------------------------------------------------
# 依赖: pip install sounddevice soundfile livekit-rtc numpy
# -------------------------------------------------------------
import threading, time, math, sys
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
import soundfile as sf
from livekit import rtc

SAMPLE_RATE = 24000          # 24 kHz
FRAME_MS    = 10             # 10 ms 一帧
FRAME_SAMP  = SAMPLE_RATE * FRAME_MS // 1000   # 240
FRAME_BYTES = FRAME_SAMP * 2                   # int16
DURATION_SEC = 10            # 录制时长

TONE_HZ     = 440            # 扬声器测试音频 (可改)
TONE_GAIN   = 0.2            # 输出音量 (0~1)

# ───────── AEC 封装 ─────────
class LocalAEC:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.frame_samp = FRAME_SAMP
        self.apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=True,
            high_pass_filter=True,
        )
        self.out_delay = 0.0
        self.in_delay  = 0.0
    def feed_render(self, pcm: bytes):
        f = rtc.AudioFrame(
            data=pcm, sample_rate=self.sr,
            num_channels=1, samples_per_channel=self.frame_samp,
        )
        self.apm.process_reverse_stream(f)
    def process_capture(self, pcm: bytes, delay_ms: int) -> bytes:
        self.apm.set_stream_delay_ms(delay_ms)
        f = rtc.AudioFrame(
            data=pcm, sample_rate=self.sr,
            num_channels=1, samples_per_channel=self.frame_samp,
        )
        self.apm.process_stream(f)
        return bytes(f.data)

# ───────── 生成一帧正弦波 ─────────
phase = 0.0
def gen_tone_frame():
    global phase
    t = (np.arange(FRAME_SAMP) + phase) / SAMPLE_RATE
    frame = (np.sin(2 * math.pi * TONE_HZ * t) * 32767 * TONE_GAIN).astype(np.int16)
    phase = (phase + FRAME_SAMP) % SAMPLE_RATE
    return frame.tobytes()

# ───────── 队列 & AEC 初始化 ─────────
q_render = Queue()   # 播放帧
aec = LocalAEC()

raw_capture  = bytearray()
clean_output = bytearray()

stop_evt = threading.Event()

# ───────── 播放回调 ─────────
def out_cb(outdata, frames, t, status):
    bytes_needed = frames * 2
    buf = bytearray()
    while len(buf) < bytes_needed:
        try:
            buf.extend(q_render.get_nowait())
        except Empty:
            buf.extend(gen_tone_frame())
    outdata[:] = np.frombuffer(buf[:bytes_needed], dtype=np.int16)
    # 分帧送 AEC
    for i in range(0, bytes_needed, FRAME_BYTES):
        aec.feed_render(buf[i:i+FRAME_BYTES])
    aec.out_delay = t.outputBufferDacTime - t.currentTime

# ───────── 录音回调 ─────────
def in_cb(indata, frames, t, status):
    aec.in_delay = t.currentTime - t.inputBufferAdcTime
    total_delay_ms = int((aec.out_delay + aec.in_delay) * 1000)
    pcm_bytes = indata.tobytes()
    for i in range(0, len(pcm_bytes), FRAME_BYTES):
        raw_chunk = pcm_bytes[i:i+FRAME_BYTES]
        clean_chunk = aec.process_capture(raw_chunk, total_delay_ms)
        raw_capture.extend(raw_chunk)
        clean_output.extend(clean_chunk)
        # 也可以实时监听：q_render.put(clean_chunk) 回放处理后音频

# ───────── 主流程 ─────────
def main():
    print("▶ 开始测试：请保持扬声器可听到 440 Hz 音，\n"
          "  并对着麦克风说话，10 秒后自动结束…")
    out_stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SAMP*4, callback=out_cb)
    in_stream  = sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SAMP*4, callback=in_cb)
    out_stream.start(); in_stream.start()

    # 预先填充队列，避免回调饿死
    for _ in range(20):
        q_render.put(gen_tone_frame())

    time.sleep(DURATION_SEC)
    stop_evt.set()
    out_stream.stop(); in_stream.stop()
    out_stream.close(); in_stream.close()

    # 保存 WAV
    sf.write("mic_raw.wav",
             np.frombuffer(raw_capture, dtype=np.int16).astype(np.float32)/32768,
             SAMPLE_RATE)
    sf.write("mic_aec.wav",
             np.frombuffer(clean_output, dtype=np.int16).astype(np.float32)/32768,
             SAMPLE_RATE)

    print("\n✅ 完成！已生成：\n"
          "   mic_raw.wav —— 原始麦克风（含回声）\n"
          "   mic_aec.wav —— AEC 处理后（回声应明显削弱）\n"
          "   现在用耳机播放对比一下效果吧。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("中断退出")
        sys.exit(0)
