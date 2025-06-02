#!/usr/bin/env python3
# listen_and_play_aec.py  ------------------------------------
# 录音 → 本地 WebRTC-AEC → (可选) 发送到服务器
# 同时接收服务器下行音频 → 播放并持续喂 render 帧
#
#   普通模式：与服务器交互
#   --dry-run ：完全本地跑通链路，验证 AEC 逻辑
#
# 依赖：
#   pip install sounddevice livekit-rtc numpy
# ------------------------------------------------------------

import argparse, socket, threading, time, logging
from queue import Queue, Empty

import sounddevice as sd
from livekit import rtc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("listen_play_aec")

# ───────────────────────── 工具 ─────────────────────────
def silence(samples: int) -> bytes:
    return b"\x00\x00" * samples        # int16 静音

# ─────────────────── AEC 封装 ──────────────────────────
class LocalAEC:
    def __init__(self, sr: int, frame_ms: int):
        self.sr      = sr
        self.samp    = sr * frame_ms // 1000
        self.byt     = self.samp * 2
        self.apm     = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=False,
            auto_gain_control=False, # 教给server
            high_pass_filter=False,
        )
        self.o_delay = 0.0
        self.i_delay = 0.0

    def feed_render(self, pcm: bytes):
        f = rtc.AudioFrame(data=pcm, sample_rate=self.sr,
                           num_channels=1, samples_per_channel=self.samp)
        self.apm.process_reverse_stream(f)

    def proc_capture(self, pcm: bytes, d_ms: int) -> bytes:
        self.apm.set_stream_delay_ms(d_ms)
        f = rtc.AudioFrame(data=pcm, sample_rate=self.sr,
                           num_channels=1, samples_per_channel=self.samp)
        self.apm.process_stream(f)
        return bytes(f.data)

# ─────────────────── 主流程 ────────────────────────────
def listen_and_play(sample_rate: int, frame_ms: int,
                    host: str, send_port: int, recv_port: int,
                    q_timeout: float, dry_run: bool):
    fs      = sample_rate * frame_ms // 1000     # 每帧采样数
    fb      = fs * 2                             # 每帧字节数
    aec     = LocalAEC(sample_rate, frame_ms)

    # socket（dry-run 时跳过）
    if not dry_run:
        sock_tx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_tx.connect((host, send_port))
        sock_rx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_rx.connect((host, recv_port))
    else:
        sock_tx = sock_rx = None
        log.info("🟡 Dry-run：不连服务器，只本地验证 AEC")

    q_tx, q_rx, stop = Queue(), Queue(), threading.Event()

    # ── 播放回调 ──
    def cb_out(outdata, frames, timing, status):
        need = frames * 2
        buf  = bytearray()
        while len(buf) < need:
            try:
                buf.extend(q_rx.get_nowait())
            except Empty:
                buf.extend(silence(fs))
        outdata[:] = buf[:need]                         # 必须 bytes
        for i in range(0, need, fb):
            aec.feed_render(buf[i:i+fb])
        aec.o_delay = timing.outputBufferDacTime - timing.currentTime

    # ── 录音回调 ──
    def cb_in(indata, frames, timing, status):
        aec.i_delay = timing.currentTime - timing.inputBufferAdcTime
        d_ms = int((aec.o_delay + aec.i_delay) * 1000)
        pcm  = bytes(indata)
        for i in range(0, len(pcm), fb):
            clean = aec.proc_capture(pcm[i:i+fb], d_ms)
            if dry_run:
                q_rx.put(clean)            # 回放处理后音
            else:
                q_tx.put(clean)

    # ── 发送 / 接收线程（dry-run 跳过） ──
    def th_send():
        while not stop.is_set():
            try:
                chunk = q_tx.get(timeout=q_timeout)
            except Empty:
                chunk = silence(fs)
            sock_tx.sendall(chunk)

    def th_recv():
        while not stop.is_set():
            chunk = sock_rx.recv(fb)
            if not chunk:
                break
            if len(chunk) < fb:
                chunk += silence(fs - len(chunk)//2)
            q_rx.put(chunk)

    # ── 启动音频流 ──
    out_stream = sd.RawOutputStream(samplerate=sample_rate, channels=1,
                                    dtype="int16", blocksize=fs*4,
                                    callback=cb_out)
    in_stream  = sd.RawInputStream (samplerate=sample_rate, channels=1,
                                    dtype="int16", blocksize=fs*4,
                                    callback=cb_in)
    out_stream.start();  in_stream.start()

    if not dry_run:
        threading.Thread(target=th_send, daemon=True).start()
        threading.Thread(target=th_recv, daemon=True).start()

    log.info("🎧 AEC 运行中%s  (Ctrl-C 退出)",
             " [dry-run]" if dry_run else "")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("⏹ 退出")
    finally:
        stop.set()
        if sock_tx: sock_tx.close();  # 为空则 dry-run
        if sock_rx: sock_rx.close()
        in_stream.stop();  out_stream.stop()
        in_stream.close(); out_stream.close()

# ─────────────────── CLI ───────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Listen + Play with local WebRTC-AEC")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--frame-ms",   type=int, default=10)
    p.add_argument("--host",       default="localhost")
    p.add_argument("--send-port",  type=int, default=12345)
    p.add_argument("--recv-port",  type=int, default=12346)
    p.add_argument("--queue-timeout", type=float, default=0.02)
    p.add_argument("--dry-run", action="store_true", help="本地 dry-run，不连服务器")
    args = p.parse_args()

    listen_and_play(sample_rate=args.sample_rate,
                    frame_ms=args.frame_ms,
                    host=args.host,
                    send_port=args.send_port,
                    recv_port=args.recv_port,
                    q_timeout=args.queue_timeout,
                    dry_run=args.dry_run)