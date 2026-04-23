from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import wave

import numpy as np

from VAD.vad_iterator import VADStateSnapshot


class VADDiagnosticsRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        sample_rate: int,
        *,
        config: dict[str, object] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.config = dict(config or {})
        self._session_counter = 0
        self._reset_session()

    def _reset_session(self) -> None:
        self._created_at: datetime | None = None
        self._audio_chunks: list[np.ndarray] = []
        self._frames: list[dict[str, object]] = []
        self._events: list[dict[str, object]] = []

    @property
    def has_data(self) -> bool:
        return bool(self._audio_chunks or self._frames or self._events)

    def _ensure_session_started(self) -> None:
        if self._created_at is None:
            self._created_at = datetime.now(timezone.utc)

    def _samples_to_ms(self, samples: int) -> float:
        return round(samples / self.sample_rate * 1000, 3)

    def record_chunk(
        self, audio_chunk: np.ndarray, snapshot: VADStateSnapshot | None
    ) -> None:
        if snapshot is None:
            return
        self._ensure_session_started()
        audio_chunk = np.asarray(audio_chunk, dtype=np.int16).copy()
        self._audio_chunks.append(audio_chunk)

        start_sample = snapshot.current_sample - snapshot.window_size_samples
        frame = asdict(snapshot)
        frame.update(
            {
                "index": len(self._frames),
                "start_sample": start_sample,
                "end_sample": snapshot.current_sample,
                "start_ms": self._samples_to_ms(start_sample),
                "end_ms": self._samples_to_ms(snapshot.current_sample),
                "duration_ms": self._samples_to_ms(snapshot.window_size_samples),
                "ending": snapshot.temp_end_sample > 0 and snapshot.triggered,
                "pre_speech_ms": self._samples_to_ms(snapshot.pre_speech_samples),
                "active_speech_ms": self._samples_to_ms(snapshot.active_speech_samples),
                "prefix_ms": self._samples_to_ms(snapshot.prefix_samples),
                "temp_end_ms": self._samples_to_ms(snapshot.temp_end_sample),
                "emitted_speech_ms": self._samples_to_ms(
                    snapshot.emitted_speech_samples
                ),
            }
        )
        self._frames.append(frame)

    def record_event(self, kind: str, timestamp_ms: float, **payload: object) -> None:
        self._ensure_session_started()
        self._events.append(
            {
                "kind": kind,
                "timestamp_ms": round(float(timestamp_ms), 3),
                **payload,
            }
        )

    def flush_session(self, reason: str) -> Path | None:
        if not self.has_data:
            return None

        self._ensure_session_started()
        session_dir = self._create_session_dir()
        audio = (
            np.concatenate(self._audio_chunks)
            if self._audio_chunks
            else np.empty(0, dtype=np.int16)
        )
        duration_ms = self._samples_to_ms(len(audio))
        session_payload = {
            "created_at": self._created_at.isoformat(),
            "flushed_at": datetime.now(timezone.utc).isoformat(),
            "flush_reason": reason,
            "sample_rate": self.sample_rate,
            "duration_ms": duration_ms,
            "chunk_count": len(self._frames),
            "event_count": len(self._events),
            "config": self.config,
            "frames": self._frames,
            "events": self._events,
        }

        self._write_audio(session_dir / "audio.wav", audio)
        (session_dir / "diagnostics.json").write_text(
            json.dumps(session_payload, indent=2),
            encoding="utf-8",
        )
        (session_dir / "report.html").write_text(
            self._build_report(session_payload, audio),
            encoding="utf-8",
        )
        self._reset_session()
        return session_dir

    def _create_session_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        base_name = f"session-{timestamp}-{self._session_counter:04d}"
        self._session_counter += 1
        session_dir = self.output_dir / base_name
        session_dir.mkdir(parents=True, exist_ok=False)
        return session_dir

    def _write_audio(self, path: Path, audio: np.ndarray) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.astype(np.int16).tobytes())

    def _build_report(self, payload: dict[str, object], audio: np.ndarray) -> str:
        frames = payload["frames"]
        events = payload["events"]
        duration_ms = max(float(payload["duration_ms"]), 1.0)
        summary_rows = [
            ("Created", payload["created_at"]),
            ("Flush reason", payload["flush_reason"]),
            ("Duration", f"{float(payload['duration_ms']):.1f} ms"),
            ("Chunks", str(payload["chunk_count"])),
            ("Events", str(payload["event_count"])),
            ("Sample rate", f"{payload['sample_rate']} Hz"),
        ]
        config_rows = [
            (key.replace("_", " "), str(value))
            for key, value in sorted(dict(payload["config"]).items())
            if value is not None
        ]
        events_rows = "".join(
            (
                "<tr>"
                f"<td>{html.escape(event['kind'])}</td>"
                f"<td>{float(event['timestamp_ms']):.1f}</td>"
                f"<td>{html.escape(self._format_event_details(event))}</td>"
                "</tr>"
            )
            for event in events
        )
        event_chips = "".join(
            (
                f'<button class="event-chip" type="button" data-seek-ms="{float(event["timestamp_ms"]):.3f}">'
                f"{html.escape(self._event_marker_label(str(event['kind'])))} "
                f"<span>{float(event['timestamp_ms']):.1f} ms</span>"
                "</button>"
            )
            for event in events
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VAD Session Diagnostics</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #1d1d1f;
      background: #fafaf8;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      margin: 0 0 16px;
    }}
    .section {{
      margin-top: 24px;
      padding: 20px;
      background: white;
      border: 1px solid #e5e5df;
      border-radius: 16px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }}
    .grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 0;
      border-bottom: 1px solid #ecece7;
      vertical-align: top;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 10px;
      font-size: 13px;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: #fcfcfb;
      border: 1px solid #ecece7;
      border-radius: 12px;
    }}
    code {{
      background: #f4f4ef;
      padding: 1px 5px;
      border-radius: 6px;
    }}
    .audio-toolbar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 10px;
      font-size: 13px;
      color: #555;
    }}
    .charts-note {{
      margin-top: 8px;
      font-size: 13px;
      color: #666;
    }}
    .event-jumps {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .event-chip {{
      border: 1px solid #d7d7cf;
      background: #f8f8f4;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    .event-chip:hover {{
      background: #efefe8;
    }}
    .event-chip span {{
      color: #666;
    }}
  </style>
</head>
<body>
  <h1>VAD Session Diagnostics</h1>
  <p>Session audio, per-frame VAD state, and the thresholds that were active while the iterator made decisions.</p>
  <div class="section">
    <h2>Audio</h2>
    <p>The raw mono recording captured by the VAD path for this session.</p>
    <audio id="session-audio" controls preload="metadata" src="audio.wav"></audio>
    <div class="audio-toolbar">
      <span id="playback-time">Playback: 0.0 ms</span>
      <span>Click a chart or event chip to seek the audio.</span>
    </div>
    <div class="event-jumps">
      {event_chips or "<span>No event markers recorded.</span>"}
    </div>
  </div>
  <div class="section">
    <h2>Summary</h2>
    <div class="grid">
      <div>
        <table>
          <tbody>
            {self._rows_to_html(summary_rows)}
          </tbody>
        </table>
      </div>
      <div>
        <table>
          <tbody>
            {self._rows_to_html(config_rows) if config_rows else "<tr><td>No custom VAD config recorded.</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="section">
    <h2>VAD Probability</h2>
    {self._build_probability_svg(frames, events, duration_ms)}
    <p class="charts-note">The red playhead follows audio playback. Click inside the chart to seek.</p>
    <div class="legend">
      <span><span class="swatch" style="background:#cdeccf"></span>Triggered speech</span>
      <span><span class="swatch" style="background:#ffe1b5"></span>End-candidate grace period</span>
      <span><span class="swatch" style="background:#b5d7ff"></span>Speech probability</span>
      <span><span class="swatch" style="background:#e57373"></span>Threshold</span>
      <span><span class="swatch" style="background:#f0a35e"></span>End threshold</span>
    </div>
  </div>
  <div class="section">
    <h2>Waveform</h2>
    {self._build_waveform_svg(audio, frames, events, duration_ms)}
  </div>
  <div class="section">
    <h2>Events</h2>
    <table>
      <thead>
        <tr><th>Kind</th><th>Time (ms)</th><th>Details</th></tr>
      </thead>
      <tbody>
        {events_rows or '<tr><td colspan="3">No VAD events recorded.</td></tr>'}
      </tbody>
    </table>
    <p>Raw frame-by-frame data is stored in <code>diagnostics.json</code>.</p>
  </div>
  <script>
    (() => {{
      const audio = document.getElementById("session-audio");
      const playbackTime = document.getElementById("playback-time");
      const charts = Array.from(document.querySelectorAll("svg[data-duration-ms]"));
      const eventButtons = Array.from(document.querySelectorAll(".event-chip[data-seek-ms]"));

      function clamp(value, min, max) {{
        return Math.min(Math.max(value, min), max);
      }}

      function formatMs(ms) {{
        return `${{ms.toFixed(1)}} ms`;
      }}

      function updatePlayheads() {{
        const currentMs = audio.currentTime * 1000;
        playbackTime.textContent = `Playback: ${{formatMs(currentMs)}}`;
        for (const svg of charts) {{
          const durationMs = parseFloat(svg.dataset.durationMs || "0");
          const left = parseFloat(svg.dataset.left || "0");
          const right = parseFloat(svg.dataset.right || "0");
          const playhead = svg.querySelector(".playhead");
          if (!playhead || durationMs <= 0 || right <= left) {{
            continue;
          }}
          const progress = clamp(currentMs / durationMs, 0, 1);
          const x = left + progress * (right - left);
          const line = playhead.querySelector("line");
          const text = playhead.querySelector("text");
          line.setAttribute("x1", x.toFixed(2));
          line.setAttribute("x2", x.toFixed(2));
          text.setAttribute("x", (x + 6).toFixed(2));
          text.textContent = formatMs(currentMs);
          playhead.setAttribute("visibility", "visible");
        }}
      }}

      function seekFromSvgEvent(svg, event) {{
        const rect = svg.getBoundingClientRect();
        const viewBox = svg.viewBox.baseVal;
        const left = parseFloat(svg.dataset.left || "0");
        const right = parseFloat(svg.dataset.right || "0");
        const durationMs = parseFloat(svg.dataset.durationMs || "0");
        if (durationMs <= 0 || right <= left || rect.width <= 0) {{
          return;
        }}
        const relativeX = (event.clientX - rect.left) / rect.width;
        const svgX = viewBox.x + relativeX * viewBox.width;
        const clampedX = clamp(svgX, left, right);
        const timeMs = ((clampedX - left) / (right - left)) * durationMs;
        audio.currentTime = timeMs / 1000;
        updatePlayheads();
      }}

      for (const svg of charts) {{
        svg.style.cursor = "pointer";
        svg.addEventListener("click", (event) => seekFromSvgEvent(svg, event));
      }}

      for (const button of eventButtons) {{
        button.addEventListener("click", () => {{
          const ms = parseFloat(button.dataset.seekMs || "0");
          audio.currentTime = ms / 1000;
          updatePlayheads();
        }});
      }}

      audio.addEventListener("timeupdate", updatePlayheads);
      audio.addEventListener("seeked", updatePlayheads);
      audio.addEventListener("loadedmetadata", updatePlayheads);
      updatePlayheads();
    }})();
  </script>
</body>
</html>
"""

    def _rows_to_html(self, rows: list[tuple[str, str]]) -> str:
        return "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
            for label, value in rows
        )

    def _format_event_details(self, event: dict[str, object]) -> str:
        details = [
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"kind", "timestamp_ms"}
        ]
        return ", ".join(details) if details else "-"

    def _event_marker_label(self, kind: str) -> str:
        labels = {
            "speech_started": "SS",
            "speech_stopped": "ST",
            "segment_emitted": "EM",
            "segment_discarded": "DD",
            "phantom_trigger": "PT",
            "session_end": "SE",
            "cleanup_flush": "CF",
        }
        return labels.get(kind, kind[:2].upper())

    def _build_probability_svg(
        self,
        frames: list[dict[str, object]],
        events: list[dict[str, object]],
        duration_ms: float,
    ) -> str:
        width = 1120
        height = 320
        left = 56
        right = 18
        top = 20
        bottom = 30
        plot_width = width - left - right
        plot_height = height - top - bottom

        def x_pos(ms: float) -> float:
            return left + (ms / duration_ms) * plot_width

        def y_pos(probability: float) -> float:
            clipped = min(max(probability, 0.0), 1.0)
            return top + (1.0 - clipped) * plot_height

        grid_lines = "".join(
            (
                f'<line x1="{left}" y1="{y_pos(level):.2f}" x2="{width - right}" '
                f'y2="{y_pos(level):.2f}" stroke="#ecece7" stroke-width="1" />'
                f'<text x="10" y="{y_pos(level) + 4:.2f}" font-size="12" fill="#666">{level:.1f}</text>'
            )
            for level in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        state_rects = self._build_state_rects(frames, x_pos, top, plot_height)
        probability_points = " ".join(
            f"{x_pos((frame['start_ms'] + frame['end_ms']) / 2):.2f},{y_pos(frame['speech_prob']):.2f}"
            for frame in frames
        )
        threshold_points = " ".join(
            f"{x_pos((frame['start_ms'] + frame['end_ms']) / 2):.2f},{y_pos(frame['threshold']):.2f}"
            for frame in frames
        )
        negative_threshold_points = " ".join(
            f"{x_pos((frame['start_ms'] + frame['end_ms']) / 2):.2f},{y_pos(frame['negative_threshold']):.2f}"
            for frame in frames
        )
        event_lines = self._build_event_lines(
            events,
            x_pos=x_pos,
            top=top,
            bottom=height - bottom,
            label_y=14,
        )
        playhead = self._build_playhead(
            left=left,
            top=top,
            bottom=height - bottom,
            label_y=top + 16,
        )
        return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="VAD probability chart" data-duration-ms="{duration_ms:.3f}" data-left="{left}" data-right="{width - right}">
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fcfcfb" />
  {grid_lines}
  {state_rects}
  <polyline fill="none" stroke="#e57373" stroke-width="2" stroke-dasharray="6 6" points="{threshold_points}" />
  <polyline fill="none" stroke="#f0a35e" stroke-width="2" stroke-dasharray="3 5" points="{negative_threshold_points}" />
  <polyline fill="none" stroke="#4d8fe6" stroke-width="2.5" points="{probability_points}" />
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#cfcfc8" />
  {event_lines}
  {playhead}
  <text x="{left}" y="{height - 8}" font-size="12" fill="#666">time (ms)</text>
  <text x="{width - right - 70}" y="{height - 8}" font-size="12" fill="#666">{duration_ms:.1f} ms</text>
</svg>
"""

    def _build_waveform_svg(
        self,
        audio: np.ndarray,
        frames: list[dict[str, object]],
        events: list[dict[str, object]],
        duration_ms: float,
    ) -> str:
        width = 1120
        height = 220
        left = 56
        right = 18
        top = 18
        bottom = 22
        plot_width = width - left - right
        plot_height = height - top - bottom
        center_y = top + plot_height / 2
        amplitude = plot_height / 2 - 4

        def x_pos(ms: float) -> float:
            return left + (ms / duration_ms) * plot_width

        state_rects = self._build_state_rects(frames, x_pos, top, plot_height)
        event_lines = self._build_event_lines(
            events,
            x_pos=x_pos,
            top=top,
            bottom=height - bottom,
            label_y=top + 12,
        )
        path_commands = self._waveform_path(
            audio, left, plot_width, center_y, amplitude
        )
        playhead = self._build_playhead(
            left=left,
            top=top,
            bottom=height - bottom,
            label_y=top + 16,
        )
        return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Audio waveform chart" data-duration-ms="{duration_ms:.3f}" data-left="{left}" data-right="{width - right}">
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fcfcfb" />
  {state_rects}
  <line x1="{left}" y1="{center_y:.2f}" x2="{width - right}" y2="{center_y:.2f}" stroke="#ecece7" stroke-width="1" />
  <path d="{path_commands}" fill="none" stroke="#303f59" stroke-width="1.2" />
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#cfcfc8" />
  {event_lines}
  {playhead}
  <text x="{left}" y="{height - 6}" font-size="12" fill="#666">recorded waveform</text>
</svg>
"""

    def _build_event_lines(
        self,
        events: list[dict[str, object]],
        *,
        x_pos,
        top: int,
        bottom: int,
        label_y: int,
    ) -> str:
        return "".join(
            (
                f'<line x1="{x_pos(float(event["timestamp_ms"])):.2f}" y1="{top}" '
                f'x2="{x_pos(float(event["timestamp_ms"])):.2f}" y2="{bottom}" '
                f'stroke="#6f6f6f" stroke-width="1" stroke-dasharray="4 4" />'
                f'<text x="{x_pos(float(event["timestamp_ms"])) + 4:.2f}" y="{label_y}" font-size="11" fill="#555">'
                f"<title>{html.escape(str(event['kind']))}</title>"
                f"{html.escape(self._event_marker_label(str(event['kind'])))}</text>"
            )
            for event in events
        )

    def _build_playhead(
        self,
        *,
        left: int,
        top: int,
        bottom: int,
        label_y: int,
    ) -> str:
        return f"""
  <g class="playhead" visibility="hidden">
    <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#d13a2f" stroke-width="2" />
    <text x="{left + 6}" y="{label_y}" font-size="11" fill="#d13a2f">0.0 ms</text>
  </g>"""

    def _build_state_rects(
        self,
        frames: list[dict[str, object]],
        x_pos,
        top: int,
        plot_height: int,
    ) -> str:
        triggered_rects = self._render_spans(
            self._spans(frames, lambda frame: bool(frame["triggered"])),
            x_pos,
            top,
            plot_height,
            fill="#cdeccf",
        )
        ending_rects = self._render_spans(
            self._spans(frames, lambda frame: bool(frame["ending"])),
            x_pos,
            top,
            plot_height,
            fill="#ffe1b5",
        )
        return ending_rects + triggered_rects

    def _render_spans(
        self,
        spans: list[tuple[float, float]],
        x_pos,
        top: int,
        plot_height: int,
        *,
        fill: str,
    ) -> str:
        return "".join(
            (
                f'<rect x="{x_pos(start_ms):.2f}" y="{top}" '
                f'width="{max(x_pos(end_ms) - x_pos(start_ms), 1.0):.2f}" '
                f'height="{plot_height}" fill="{fill}" opacity="0.65" />'
            )
            for start_ms, end_ms in spans
        )

    def _spans(
        self,
        frames: list[dict[str, object]],
        predicate,
    ) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        current_start: float | None = None
        previous_end = 0.0
        for frame in frames:
            active = predicate(frame)
            start_ms = float(frame["start_ms"])
            end_ms = float(frame["end_ms"])
            if active and current_start is None:
                current_start = start_ms
            if not active and current_start is not None:
                spans.append((current_start, previous_end))
                current_start = None
            previous_end = end_ms
        if current_start is not None:
            spans.append((current_start, previous_end))
        return spans

    def _waveform_path(
        self,
        audio: np.ndarray,
        left: int,
        plot_width: int,
        center_y: float,
        amplitude: float,
    ) -> str:
        if len(audio) == 0:
            return f"M {left} {center_y:.2f} L {left + plot_width} {center_y:.2f}"

        points = min(plot_width, len(audio))
        edges = np.linspace(0, len(audio), num=points + 1, dtype=int)
        commands: list[str] = []
        for index in range(points):
            segment = audio[edges[index] : edges[index + 1]]
            if len(segment) == 0:
                continue
            x = left + (index / max(points - 1, 1)) * plot_width
            min_value = float(segment.min()) / 32768.0
            max_value = float(segment.max()) / 32768.0
            y_min = center_y - max_value * amplitude
            y_max = center_y - min_value * amplitude
            commands.append(f"M {x:.2f} {y_min:.2f} L {x:.2f} {y_max:.2f}")
        return " ".join(commands)
