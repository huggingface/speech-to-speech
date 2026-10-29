"""Bounded, cancellable OpenAI-compatible SSE inference, without provider SDKs."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

import httpx


class InferenceError(RuntimeError):
    pass


class VisibleTextFilter:
    """Remove thought blocks and control tokens, even across chunk boundaries.

    Gemma emits an empty thought channel even with thinking disabled. Never
    forward reasoning_content/reasoning fields; this also guards unparsed
    thought channels in content. An unfinished thought block fails closed.
    """

    def __init__(self):
        self.pending = ""
        self.block_end = None

    def feed(self, text):
        self.pending += text
        out = []
        while self.pending:
            if self.block_end:
                at = self.pending.find(self.block_end)
                if at < 0:
                    self.pending = self.pending[-len(self.block_end) :]
                    break
                self.pending = self.pending[at + len(self.block_end) :]
                self.block_end = None
                continue
            at = self.pending.find("<")
            if at < 0:
                out.append(self.pending)
                self.pending = ""
                break
            out.append(self.pending[:at])
            self.pending = self.pending[at:]
            end = self.pending.find(">")
            if end < 0:
                if len(self.pending) > 256:
                    raise InferenceError("Unterminated inference control token")
                break
            tag = self.pending[: end + 1]
            if tag == "<|channel>":
                newline = self.pending.find("\n", end)
                if newline < 0:
                    if len(self.pending) > 256:
                        raise InferenceError("Unterminated inference channel")
                    break
                channel = self.pending[end + 1 : newline].strip()
                self.pending = self.pending[newline + 1 :]
                if channel in ("thought", "analysis"):
                    self.block_end = "<channel|>"
                continue
            self.pending = self.pending[end + 1 :]
            if tag == "<think>":
                self.block_end = "</think>"
            # All other serialized control tokens are intentionally dropped.
        return "".join(out)


class RemoteInference:
    def __init__(self, config, *, transport=None):
        self.config = dict(config)
        self.base_url = os.environ.get(config.get("base_url_env", ""), config.get("base_url", "")).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Remote inference requires an explicit HTTP(S) base_url")
        if httpx.URL(self.base_url).userinfo or httpx.URL(self.base_url).query:
            raise ValueError("Put inference credentials in api_key_env, not the endpoint URL")
        self.model = os.environ.get(config.get("model_env", ""), config.get("model", ""))
        if not self.model:
            raise ValueError("Remote inference requires a model")
        self.api_key = os.environ.get(config.get("api_key_env") or "", "")
        self.connect_timeout = float(config.get("connect_timeout_s", 10))
        self.read_timeout = float(config.get("read_timeout_s", 15))
        self.request_timeout = float(config.get("request_timeout_s", 60))
        if min(self.connect_timeout, self.read_timeout, self.request_timeout) <= 0:
            raise ValueError("Inference timeouts must be positive")
        self.cancel_event = threading.Event()
        self.transport = transport
        self._lock = threading.Lock()
        self._active = set()

    def configure_request(self, *, instructions=""):
        self.context_instructions = instructions or ""

    def begin_turn(self, cancel_event):
        self.cancel_event = cancel_event

    def cancel(self):
        self.cancel_event.set()
        with self._lock:
            active = list(self._active)
        # Closing from another thread interrupts a blocked socket read. The
        # consumer itself never waits on that read or on a provider SDK.
        for response in active:
            threading.Thread(target=response.close, daemon=True).start()

    def stream(self, route, payload):
        cancel = self.cancel_event  # captured per turn; never clear/reuse it
        deadline = time.monotonic() + self.request_timeout
        events = queue.Queue(maxsize=64)
        stopped = threading.Event()
        owned_responses = []
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}

        def put(event):
            while not cancel.is_set() and not stopped.is_set():
                try:
                    events.put(event, timeout=0.05)
                    return
                except queue.Full:
                    continue

        def read():
            try:
                timeout = httpx.Timeout(self.read_timeout, connect=self.connect_timeout)
                with httpx.Client(
                    timeout=timeout, transport=self.transport or httpx.HTTPTransport(retries=2)
                ) as client:
                    if cancel.is_set() or stopped.is_set():
                        return
                    with client.stream("POST", self.base_url + route, headers=headers, json=payload) as response:
                        if cancel.is_set() or stopped.is_set():
                            return
                        with self._lock:
                            self._active.add(response)
                            owned_responses.append(response)
                        try:
                            if response.status_code != 200:
                                raise InferenceError(f"Inference HTTP {response.status_code}")
                            data = []
                            for line in response.iter_lines():
                                if cancel.is_set() or stopped.is_set():
                                    return
                                if line.startswith("data:"):
                                    data.append(line[5:].lstrip())
                                elif not line and data:
                                    raw = "\n".join(data)
                                    data = []
                                    if raw == "[DONE]":
                                        put(("done", None))
                                        return
                                    event = json.loads(raw)
                                    if "error" in event:
                                        raise InferenceError("Inference stream reported an error")
                                    put(("chunk", event))
                            # A broken connection must not commit buffered text.
                            raise InferenceError("Inference stream ended without [DONE]")
                        finally:
                            with self._lock:
                                self._active.discard(response)
            except Exception as exc:
                error = (
                    exc
                    if isinstance(exc, InferenceError)
                    else InferenceError(f"Inference stream failed ({type(exc).__name__})")
                )
                put(("error", error))

        worker = threading.Thread(target=read, daemon=True, name="convfill-http-stream")
        worker.start()
        try:
            while not cancel.is_set():
                if time.monotonic() >= deadline:
                    raise InferenceError("Inference request deadline exceeded")
                try:
                    kind, value = events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if kind == "done":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            stopped.set()
            with self._lock:
                active = list(owned_responses)
            for response in active:
                threading.Thread(target=response.close, daemon=True).start()
            worker.join(timeout=0.2)

    def knowledge(self, prompt):
        from .conversational_stream_parser import ConversationalStreamParser

        cancel = self.cancel_event
        parser, visible = ConversationalStreamParser(), VisibleTextFilter()
        thinking = bool(self.config.get("enable_thinking", False))
        # Gemma's template controls thinking with this system token. The
        # default explicitly omits it, as well as sending enable_thinking=false.
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": ("<|think|>" if thinking else "")
                    + "Supply concise factual knowledge sentences. Do not include internal reasoning.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": int(self.config.get("max_tokens", 256)),
            "temperature": float(self.config.get("temperature", 1.0)),
            "top_p": float(self.config.get("top_p", 0.95)),
            "top_k": int(self.config.get("top_k", 64)),
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        payload["messages"][0]["content"] += "\n" + getattr(self, "context_instructions", "")
        payload["messages"][0]["content"] += (
            "\nYou supply content to a separate Talker, which already handles acknowledgments and empathy. "
            "Supply only new useful substance or a necessary follow-up question. "
            "Do not greet, apologize, acknowledge, announce a search, or repeat the Talker's words. "
            "If you cannot establish that a current fact is fresh, say so instead of inventing a result. "
            "Today's date is " + datetime.now(timezone.utc).date().isoformat() + "."
        )
        seen_text = False
        for event in self.stream("/chat/completions", payload):
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                safe_text = visible.feed(text)
                seen_text = seen_text or bool(safe_text.strip())
                yield from parser.feed(safe_text)
        if not cancel.is_set():
            if not seen_text:
                raise InferenceError("Reasoner returned no visible knowledge")
            yield from parser.flush()

    def completion(self, token_ids, stop_ids):
        payload = {
            "model": self.model,
            "prompt": token_ids,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": int(self.config.get("max_tokens", 256)),
            "temperature": float(self.config.get("temperature", 0)),
            "stop_token_ids": stop_ids,
            "skip_special_tokens": False,
            "include_stop_str_in_output": False,
            "echo": False,
        }
        count = 0
        for event in self.stream("/completions", payload):
            if event.get("usage"):
                count = int(event["usage"].get("completion_tokens", count))
                yield "", count
            for choice in event.get("choices", []):
                text = choice.get("text") or ""
                count += bool(text)
                yield text, count
