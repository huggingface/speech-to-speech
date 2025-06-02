#!/usr/bin/env python3
# listen_and_play_safe.py —— 16 kHz + WebRTC AEC + 队列缺包保险
# ---------------------------------------------------------------
# pip install sounddevice transformers livekit-rtc
# livekit-rtc ≥ 1.19.0
# ---------------------------------------------------------------
import socket, threading
from queue import Queue, Empty
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser
from livekit import rtc

# ───────────── CLI 参数 ─────────────
@dataclass
class ListenAndPlayArguments:
    send_rate:   int  = field(default=16000, metadata={"help": "固定 16 kHz"})
    recv_rate:   int  = field(default=16000, metadata={"help": "固定 16 kHz"})
    chunk_bytes: int  = field(default=320,   metadata={"help": "10 ms 帧 (160 sample × 16-bit)"})
    host:        str  = field(default="localhost")
    send_port:   int  = field(default=12345)
    recv_port:   int  = field(default=12346)
    dry_run:     bool = field(default=False, metadata={"help": "本地回环，不连服务器"})

# ───────────── 主流程 ─────────────
def listen_and_play(send_rate   = 16000,
                    recv_rate   = 16000,
                    chunk_bytes = 320,
                    host        = "localhost",
                    send_port   = 12345,
                    recv_port   = 12346,
                    dry_run     = False):

    BYTES_PER_SAMPLE = 2
    chunk_frames     = chunk_bytes // BYTES_PER_SAMPLE        # 160
    samples_per_chan = chunk_frames
    print(f"▶ 每帧 {chunk_frames} 样本 / {chunk_bytes} 字节  (16 kHz, 10 ms)")

    # ———— 网络连接 ————
    if not dry_run:
        sock_tx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_tx.connect((host, send_port))
        sock_rx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_rx.connect((host, recv_port))
    else:
        sock_tx = sock_rx = None
        print("🟡 Dry-run：不与服务器通信，只本地回放")

    # ———— 队列 / 状态 ————
    q_tx, q_rx       = Queue(), Queue()
    stop_evt         = threading.Event()
    last_frame       = bytearray(b"\x00" * chunk_bytes)   # 初始静音
    shortage_counter = 0
    MAX_REPEAT       = 3   # ≤ 30 ms 使用 last_frame，否则静音
    play_delay_s     = 0.0
    rec_delay_s      = 0.0

    # ———— APM (AEC) ————
    apm = rtc.AudioProcessingModule(
        echo_cancellation=True,
        noise_suppression=False,
        auto_gain_control=False,
        high_pass_filter=False,
    )

    # ———— 播放回调 ————
    def cb_out(outdata, frames, timing, status):
        nonlocal play_delay_s, last_frame, shortage_counter

        need = frames * BYTES_PER_SAMPLE
        buf  = bytearray()

        while len(buf) < need:
            try:
                pkt = q_rx.get_nowait()
                buf.extend(pkt)
                last_frame[:]     = pkt[:chunk_bytes]
                shortage_counter  = 0                    # 来包就复位
            except Empty:
                shortage_counter += 1
                if shortage_counter <= MAX_REPEAT:
                    buf.extend(last_frame)               # 短缺：重播上一帧
                else:
                    buf.extend(b"\x00" * chunk_bytes)    # 长缺：静音

        # ★ 先送 render 帧给 AEC
        for i in range(0, need, chunk_bytes):
            frame = rtc.AudioFrame(
                data=buf[i:i+chunk_bytes],
                sample_rate=recv_rate,
                num_channels=1,
                samples_per_channel=samples_per_chan,
            )
            apm.process_reverse_stream(frame)

        # 然后写扬声器
        outdata[:need] = buf[:need]

        # 记录播放路径延迟
        play_delay_s = timing.outputBufferDacTime - timing.currentTime

    # ———— 录音回调 ————
    def cb_in(indata, frames, timing, status):
        nonlocal rec_delay_s, play_delay_s

        rec_delay_s = timing.currentTime - timing.inputBufferAdcTime
        total_delay_ms = int((play_delay_s + rec_delay_s) * 1000)

        pcm = bytes(indata)
        for i in range(0, len(pcm), chunk_bytes):
            piece = pcm[i:i+chunk_bytes]
            frame = rtc.AudioFrame(
                data=piece,
                sample_rate=send_rate,
                num_channels=1,
                samples_per_channel=samples_per_chan,
            )
            apm.set_stream_delay_ms(total_delay_ms)
            apm.process_stream(frame)
            cleaned = bytes(frame.data)

            if dry_run:
                q_rx.put(cleaned)
            else:
                q_tx.put(cleaned)

    # ———— 发送 / 接收线程 ————
    def th_send():
        while not stop_evt.is_set():
            data = q_tx.get()
            if sock_tx:
                sock_tx.sendall(data)

    def th_recv():
        buf = bytearray()
        while not stop_evt.is_set():
            pkt = sock_rx.recv(4096)
            if not pkt:
                break
            buf.extend(pkt)
            while len(buf) >= chunk_bytes:
                q_rx.put(bytes(buf[:chunk_bytes]))
                del buf[:chunk_bytes]

    # ———— 启动音频 ————
    in_stream  = sd.RawInputStream(
        samplerate=send_rate, channels=1, dtype="int16",
        blocksize=chunk_frames*4, callback=cb_in)
    out_stream = sd.RawOutputStream(
        samplerate=recv_rate, channels=1, dtype="int16",
        blocksize=chunk_frames*4, callback=cb_out)

    in_stream.start(); out_stream.start()

    if not dry_run:
        threading.Thread(target=th_send, daemon=True).start()
        threading.Thread(target=th_recv, daemon=True).start()

    try:
        input("Recording & streaming… （按 Enter 停止）\n")
    finally:
        stop_evt.set()
        for s in (sock_tx, sock_rx):
            if s: s.close()
        in_stream.stop(); out_stream.stop()
        in_stream.close(); out_stream.close()
        print("Connection closed.")

# ───────────── CLI 入口 ─────────────
if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    args.send_rate = args.recv_rate = 16000    # 强制 16 kHz
    listen_and_play(**vars(args))