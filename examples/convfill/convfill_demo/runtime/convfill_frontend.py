import json
import os
import re
import threading
import time

from transformers import AutoTokenizer

from convfill_demo.runtime.turn_state_manager import TurnStateManager

_SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+")


def _flush_sentences(buf: str) -> tuple[list[str], str]:
    parts = _SENTENCE_SPLIT_RE.split(buf)
    if len(parts) == 1:
        return [], buf
    return [p for p in parts[:-1] if p.strip()], parts[-1]


def _without_repeated_sentence_prefix(text: str, previous: str) -> str:
    """Drop exact sentence echoes at a streaming handoff, never paraphrases."""

    def key(value):
        return re.findall(r"\w+(?:'\w+)*", value.replace("’", "'").casefold())

    prior = [part.strip() for part in _SENTENCE_SPLIT_RE.split(previous.strip()) if part.strip()]
    incoming = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]
    # Only a complete sentence with at least three words is an unambiguous echo.
    while prior and incoming and len(key(incoming[0])) >= 3 and key(prior[-1]) == key(incoming[0]):
        incoming.pop(0)
    return " ".join(incoming)


class ConvFillFrontend:
    """The fine-tuned Talker, served by vLLM through raw completions."""

    def __init__(self, model_config_path, dialogue_state_manager):
        self.model_config_path = model_config_path

        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        self.model_config = model_config

        self.boundary_tokens = self.model_config["boundary_tokens"]
        self.model_name = self.model_config["model_name"]
        if self.model_config.get("backend") != "vllm":
            raise ValueError(f"{model_config_path} must use the vllm Talker backend")

        self.cancel_event = threading.Event()
        self.remote = None
        self.tokenizer = self.load_tokenizer()
        self.turn_state_manager = TurnStateManager(boundary_tokens=self.boundary_tokens)
        self.dialogue_state_manager = dialogue_state_manager
        self.last_generate_ms: float | None = None
        self.last_generated_tokens: int | None = None
        self.last_first_token_ms: float | None = None
        self.last_first_sentence_ms: float | None = None

    def get_boundary_tokens(self):
        return self.boundary_tokens

    def load_tokenizer(self):
        from .remote_inference import RemoteInference

        checkpoint = self.model_config["convfill_hf_pretrained_path"]
        revision = self.model_config.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        self._validate_remote_tokenizer(tokenizer)
        self.remote = RemoteInference({"model": checkpoint, **self.model_config.get("remote", {})})
        return tokenizer

    def _validate_remote_tokenizer(self, tokenizer):
        expected = self.model_config.get("expected_token_ids", {})
        for token in ["<sil>", self.boundary_tokens["END"].strip(), "<|im_start|>"]:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids) != token:
                raise ValueError(f"Fine-tuned checkpoint does not preserve {token!r} as one token")
            if token in expected and ids[0] != expected[token]:
                raise ValueError(f"Fine-tuned token ID mismatch for {token!r}: {ids}")

    def begin_turn(self, cancel_event):
        self.cancel_event = cancel_event
        if self.remote:
            self.remote.begin_turn(cancel_event)

    def cancel(self):
        self.cancel_event.set()
        if self.remote:
            self.remote.cancel()

    def _iter_remote_chunks(self, prompt):
        # Send exact IDs from the fine-tuned tokenizer: no chat template, no
        # added BOS, no re-tokenization by an unrelated base checkpoint.
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        stop_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.encode(self.boundary_tokens["END"].strip(), add_special_tokens=False)[0],
        ]
        yield from self.remote.completion(ids, sorted(set(stop_ids)))

    def reset_turn_state(self):
        self.turn_state_manager.reset()

    def run_convfill_inference(self, prompt):
        end_tok = self.boundary_tokens["END"].strip()
        chunk_iter = self._iter_remote_chunks(prompt)

        t0 = time.perf_counter()
        self.last_first_token_ms = None
        self.last_first_sentence_ms = None

        from .remote_inference import VisibleTextFilter

        visible = VisibleTextFilter()
        buf = ""
        raw_pending = ""
        last_token_count = 0
        hit_end = False
        try:
            for chunk, token_count in chunk_iter:
                if self.cancel_event.is_set():
                    break
                last_token_count = token_count
                if not chunk:
                    continue
                if self.last_first_token_ms is None:
                    self.last_first_token_ms = (time.perf_counter() - t0) * 1000.0
                raw_pending += chunk
                if end_tok and end_tok in raw_pending:
                    safe_chunk = raw_pending.split(end_tok, 1)[0]
                    raw_pending = ""
                    hit_end = True
                else:
                    # Keep a possible partial END token until the next chunk.
                    hold = 0
                    for length in range(1, len(end_tok)):
                        if raw_pending.endswith(end_tok[:length]):
                            hold = length
                    safe_chunk = raw_pending[:-hold] if hold else raw_pending
                    raw_pending = raw_pending[-hold:] if hold else ""
                buf += visible.feed(safe_chunk)

                sentences, buf = _flush_sentences(buf)
                for s in sentences:
                    s = s.lstrip()
                    if s:
                        if self.last_first_sentence_ms is None:
                            self.last_first_sentence_ms = (time.perf_counter() - t0) * 1000.0
                        yield s

                if hit_end:
                    break
        finally:
            chunk_iter.close()

        self.last_generate_ms = (time.perf_counter() - t0) * 1000.0
        self.last_generated_tokens = last_token_count

        tail = buf.strip()
        if tail and not self.cancel_event.is_set():
            if self.last_first_sentence_ms is None:
                self.last_first_sentence_ms = (time.perf_counter() - t0) * 1000.0
            yield tail

    def infer(self, current_user_input, current_thought, history=None):
        assert history is not None, "History must be provided for inference."
        if os.environ.get("CONVFILL_LOG_CONTENT") == "1":
            print("[Inferring with thought] " + current_thought, flush=True)
        prompt = self.turn_state_manager.build_prompt(
            current_user_input=current_user_input,
            current_thought=current_thought,
            history=history,
        )
        yield from self.run_convfill_inference(prompt)

    def infer_and_update_state(self, current_thought):
        history = self.dialogue_state_manager.get_history()
        user_in = self.dialogue_state_manager.user_turns[-1]
        pieces: list[str] = []
        for fragment in self.infer(
            current_user_input=user_in,
            current_thought=current_thought,
            history=history,
        ):
            previous = " ".join(self.turn_state_manager.responses + pieces)
            fragment = _without_repeated_sentence_prefix(fragment, previous)
            if fragment:
                pieces.append(fragment)
                yield fragment
        if self.cancel_event.is_set():
            return
        full_response = " ".join(pieces).strip()
        if self.turn_state_manager.user_turn_empty():
            self.turn_state_manager.add_user_turn(user_in)
        if full_response:
            self.turn_state_manager.add_response(full_response)
        self.turn_state_manager.add_thought(current_thought)

    def get_final_response_and_reset_turn_state(self):
        user_turn, final_response, final_thoughts = self.turn_state_manager.get_completed_response()
        self.dialogue_state_manager.update_response_state(final_response)
        self.dialogue_state_manager.update_thoughts_state(self.turn_state_manager.thoughts)
        self.reset_turn_state()
        return user_turn, final_response, final_thoughts
