import json
import logging
import os
import queue
import re
import threading
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from convfill_demo.runtime.turn_state_manager import TurnStateManager

_SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+")


def _flush_sentences(buf: str) -> tuple[list[str], str]:
    parts = _SENTENCE_SPLIT_RE.split(buf)
    if len(parts) == 1:
        return [], buf
    return [p for p in parts[:-1] if p.strip()], parts[-1]


# remove faulty Gemma error
logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)

# proceess-wide cache
_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.Lock()

_DTYPE_NAMES = {
    "bf16": torch.bfloat16,
}


class ConvFillFrontend:
    def __init__(self, model_config_path, dialogue_state_manager, device: str = "cpu", dtype: str | None = None):
        self.model_config_path = model_config_path
        self.device = device
        self.dtype = dtype

        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        self.model_config = model_config

        self.boundary_tokens = self.model_config["boundary_tokens"]
        self.special_tokens = self.model_config["special_tokens"]
        self.add_special_tokens = self.model_config["add_special_tokens"]
        self.model_name = self.model_config["model_name"]

        # backend defaults to mlx
        self.backend = self.model_config.get("backend", "mlx")

        self.cancel_event = threading.Event()
        self.remote = None
        self.model, self.tokenizer = self.load_model()
        self.turn_state_manager = TurnStateManager(boundary_tokens=self.boundary_tokens)
        self.dialogue_state_manager = dialogue_state_manager
        self.last_generate_ms: float | None = None
        self.last_generated_tokens: int | None = None
        self.last_first_token_ms: float | None = None
        self.last_first_sentence_ms: float | None = None

    def get_boundary_tokens(self):
        return self.boundary_tokens

    def load_model(self):
        if self.backend == "vllm":
            from .remote_inference import RemoteInference

            checkpoint = self.model_config["convfill_hf_pretrained_path"]
            revision = self.model_config.get("revision", "main")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
            self._validate_remote_tokenizer(tokenizer)
            self.remote = RemoteInference({"model": checkpoint, **self.model_config.get("remote", {})})
            return None, tokenizer
        if self.backend == "mlx":
            model_checkpoint_path = self.model_config.get("convfill_hf_mlx_pretrained_path")
        else:
            model_checkpoint_path = self.model_config.get("convfill_hf_pretrained_path")

        if not model_checkpoint_path:
            field_name = "convfill_hf_mlx_pretrained_path" if self.backend == "mlx" else "convfill_hf_pretrained_path"
            raise ValueError(f"{self.model_config_path} is missing {field_name}")

        config_dtype = self.model_config.get("torch_dtype")
        if self.dtype is not None:
            resolved_dtype = _DTYPE_NAMES[self.dtype]
        elif config_dtype is not None:
            resolved_dtype = _DTYPE_NAMES[config_dtype]
        else:
            resolved_dtype = torch.bfloat16

        cache_key = (
            model_checkpoint_path,
            self.backend,
            bool(self.add_special_tokens),
            json.dumps(self.special_tokens, sort_keys=True) if self.special_tokens else "",
            self.device,
            str(resolved_dtype),
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                model, tokenizer = cached
                print("[MODEL SETUP] REUSING CACHED MODEL", self.model_name, flush=True)
                return model, tokenizer

            if self.backend == "mlx":
                model, tokenizer = self._load_mlx(model_checkpoint_path)
            else:
                model, tokenizer = self._load_hf(model_checkpoint_path, resolved_dtype)

            _MODEL_CACHE[cache_key] = (model, tokenizer)
            return model, tokenizer

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

    def _load_hf(self, model_checkpoint_path, resolved_dtype):
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint_path,
            torch_dtype=resolved_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        print("[MODEL SETUP] LOADED HF MODEL", self.model_name, "dtype:", resolved_dtype, flush=True)
        print("[MODEL SETUP] LOADED WEIGHTS FROM", model_checkpoint_path, flush=True)

        current_vocab = model.get_input_embeddings().weight.shape[0]

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        print("[MODEL SETUP] PAD TOKEN ID:", tokenizer.pad_token_id, flush=True)

        if self.add_special_tokens:
            tokenizer.add_special_tokens(self.special_tokens)
        print("[MODEL SETUP] ADDED SPECIAL TOKENS:", self.special_tokens, flush=True)

        if len(tokenizer) != current_vocab:
            model.resize_token_embeddings(len(tokenizer))
            model.tie_weights()

        model.to(self.device)
        model.tie_weights()
        model.eval()
        print(f"[MODEL SETUP] MOVED MODEL TO DEVICE: {self.device}", flush=True)

        # warmup
        with torch.no_grad():
            warmup_ids = tokenizer("hello", return_tensors="pt")["input_ids"].to(self.device)
            model.generate(
                input_ids=warmup_ids,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("[MODEL SETUP] WARMED UP", self.model_name, flush=True)
        return model, tokenizer

    def _load_mlx(self, model_checkpoint_path):
        import mlx_lm

        model, tokenizer = mlx_lm.load(model_checkpoint_path)
        print("[MODEL SETUP] LOADED MLX MODEL", self.model_name, flush=True)
        print("[MODEL SETUP] LOADED WEIGHTS FROM", model_checkpoint_path, flush=True)

        if self.add_special_tokens:
            for tok in self.special_tokens.get("additional_special_tokens", []):
                ids = tokenizer.encode(tok, add_special_tokens=False)
                assert len(ids) == 1, (
                    f"Special token {tok!r} did not survive MLX conversion as a "
                    f"single token id (got {ids}). Re-run convert_to_mlx_q8 or "
                    f"check the source checkpoint's tokenizer."
                )

        # register the END boundary as an extra stop so generation halts
        end_tok = self.boundary_tokens["END"].strip()
        try:
            tokenizer.add_eos_token(end_tok)
        except Exception as e:
            print(f"[MODEL SETUP] WARNING: could not register extra EOS {end_tok!r}: {e}", flush=True)

        # warmup
        _ = mlx_lm.generate(model, tokenizer, prompt="hello", max_tokens=4, verbose=False)
        print("[MODEL SETUP] WARMED UP", self.model_name, flush=True)
        return model, tokenizer

    def reset_turn_state(self):
        self.turn_state_manager.reset()

    def get_thoughts_and_reset_turn_state(self):
        """Return this turn's consumed thoughts and drop the partial turn.

        Used when a turn is interrupted: the response the user heard is decided
        by the transport, but the Reasoner knowledge that produced it is only
        known here.
        """
        thoughts = list(self.turn_state_manager.thoughts)
        self.reset_turn_state()
        return thoughts

    def _iter_hf_chunks(self, prompt):
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = encoded["input_ids"].to(self.device)
        pad_id = self.tokenizer.pad_token_id

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            timeout=1.0,
        )
        cancel = self.cancel_event
        generation_stop = threading.Event()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set() or generation_stop.is_set()

        gen_kwargs = dict(
            stopping_criteria=StoppingCriteriaList([Cancelled()]),
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=False,
            top_p=None,
            top_k=None,
            pad_token_id=pad_id,
            streamer=streamer,
        )

        gen_error: list[BaseException] = []

        def _run_generate():
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except BaseException as exc:
                gen_error.append(exc)
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        try:
            count = 0
            while not cancel.is_set():
                try:
                    chunk = next(streamer)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue
                except StopIteration:
                    break
                if chunk:
                    count += 1
                yield chunk, count
        finally:
            generation_stop.set()
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise TimeoutError("Transformers Talker did not stop")
            if gen_error:
                raise gen_error[0]

    def _iter_mlx_chunks(self, prompt):
        import mlx_lm

        n = 0
        for resp in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=256):
            if self.cancel_event.is_set():
                break
            n = resp.generation_tokens
            yield resp.text, n

    def run_convfill_inference(self, prompt):
        end_tok = self.boundary_tokens["END"].strip()

        if self.backend == "vllm":
            chunk_iter = self._iter_remote_chunks(prompt)
        elif self.backend == "mlx":
            chunk_iter = self._iter_mlx_chunks(prompt)
        else:
            chunk_iter = self._iter_hf_chunks(prompt)

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
            pieces.append(fragment)
            yield fragment
        if self.cancel_event.is_set():
            return
        full_response = " ".join(pieces).strip()
        if self.turn_state_manager.user_turn_empty():
            self.turn_state_manager.add_user_turn(user_in)
        self.turn_state_manager.add_response(full_response)
        self.turn_state_manager.add_thought(current_thought)

    def get_final_response_and_reset_turn_state(self):
        user_turn, final_response, final_thoughts = self.turn_state_manager.get_completed_response()
        self.dialogue_state_manager.update_response_state(final_response)
        self.dialogue_state_manager.update_thoughts_state(self.turn_state_manager.thoughts)
        self.reset_turn_state()
        return user_turn, final_response, final_thoughts
