from baseHandler import BaseHandler
from livekit import rtc

FRAME_BYTES = 320         # 10 ms @ 16kHz mono PCM, 16-bit = 2 bytes/sample
MIN_OUTPUT_BYTES = 1024   # 最小输出块：64 ms / 512 samples

class LivekitAecHandler(BaseHandler):
    def setup(self):
        self.apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=True,
        )
        self._capture_buf = bytearray()
        self._render_buf = bytearray()
        self._out_buf = bytearray()  # 用于累积已处理后的帧（送到 VAD）

    def _bytes_to_frame(self, pcm: bytes, sample_rate: int = 16000, num_channels: int = 1) -> rtc.AudioFrame:
        samples_per_channel = len(pcm) // 2 // num_channels
        return rtc.AudioFrame(
            data=pcm,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )

    def _slice_into_frames(self, buf: bytearray, chunk: bytes):
        """
        将任意长度音频按 320B/帧 切片，不足部分保留到 buf。
        """
        buf.extend(chunk)
        while len(buf) >= FRAME_BYTES:
            yield bytes(buf[:FRAME_BYTES])
            del buf[:FRAME_BYTES]

    def process(self, capture_chunk: bytes, render_chunk: bytes | None = None):
        """
        主处理方法：返回 1024B 为单位的 AEC 后 PCM 数据块。
        """
        # 1) 可选：先处理 render 参考
        if render_chunk:
            for r in self._slice_into_frames(self._render_buf, render_chunk):
                self.apm.process_reverse_stream(self._bytes_to_frame(r))

        # 2) capture 侧帧处理并缓存结果
        for c in self._slice_into_frames(self._capture_buf, capture_chunk):
            frame = self._bytes_to_frame(c)
            self.apm.process_stream(frame)
            self._out_buf.extend(bytes(frame.data))

        # 3) 一次性输出 ≥ 1024B 的数据块
        while len(self._out_buf) >= MIN_OUTPUT_BYTES:
            yield bytes(self._out_buf[:MIN_OUTPUT_BYTES])
            del self._out_buf[:MIN_OUTPUT_BYTES]

    def flush(self):
        """
        返回处理完的最后一段数据（已补零到 1024B），保证下游模块不再抛异常。
        """
        flushed = bytearray()

        # 1) 把 capture_buf 中的残帧也处理掉
        if len(self._capture_buf) >= FRAME_BYTES:
            for c in self._slice_into_frames(self._capture_buf, b""):
                frame = self._bytes_to_frame(c)
                self.apm.process_stream(frame)
                self._out_buf.extend(bytes(frame.data))

        # 2) 如果还有未输出的结果，补齐到 MIN_OUTPUT_BYTES
        if self._out_buf:
            if len(self._out_buf) < MIN_OUTPUT_BYTES:
                pad_len = MIN_OUTPUT_BYTES - len(self._out_buf)
                self._out_buf.extend(b"\x00" * pad_len)
            flushed = self._out_buf
            self._out_buf = bytearray()  # 清空缓冲区

        return bytes(flushed) if flushed else None
