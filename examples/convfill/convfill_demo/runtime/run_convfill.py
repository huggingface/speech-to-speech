# IMPORTANT: Install the package first with `pip install -e .` from the repo root

import json
import queue
import threading
import time
from pathlib import Path

from strip_markdown import strip_markdown

from convfill_demo.runtime.convfill_backend_multi import ConvFillBackend
from convfill_demo.runtime.convfill_frontend import ConvFillFrontend
from convfill_demo.runtime.dialogue_state_manager import DialogueStateManager
from convfill_demo.runtime.model_inference_functions_conversational import BackendInference


class ConvFillConfig:
    def __init__(self, config_path, mode=None):
        example_root = Path(__file__).resolve().parents[2]
        config_file = Path(config_path)
        if not config_file.is_absolute() and not config_file.is_file():
            config_file = example_root / config_file
        with config_file.open() as f:
            config = json.load(f)

        if "modes" in config:
            mode_config = config["modes"][mode]
            config = {**config, **mode_config, "mode": mode}

        def resolve_resource(value):
            resource = Path(value)
            return str(resource if resource.is_absolute() else example_root / resource)

        self.frontend_model_config_path = resolve_resource(config["frontend_model_config_path"])
        self.backend_prompt_template_file = resolve_resource(config["backend_prompt_template_file"])
        self.backend_model_name = config["backend_model_name"]
        self.backend_model_mode = config["backend_model_mode"]
        self.num_history_turns = config["num_history_turns"]
        self.tts_mode = config["tts_mode"]
        self.tts_model_path = config.get("tts_model_path")
        self.task_specific_config = config.get("task_specific_config", None)
        self.mode = config["mode"]
        self.frontend_device = config.get("frontend_device", "cpu")
        self.frontend_dtype = config.get("frontend_dtype")
        self.reasoner = config.get("reasoner", {})
        self.turn_timeout_s = float(config.get("turn_timeout_s", 90))
        self.shutdown_timeout_s = float(config.get("shutdown_timeout_s", 3))
        if self.turn_timeout_s <= 0 or self.shutdown_timeout_s <= 0:
            raise ValueError("Turn timeouts must be positive")


class ConvFillSystem:
    def __init__(
        self,
        convfill_config,
        dialogue_state_manager=None,
        on_thought=None,
        on_response_fragment=None,
        on_rag_context=None,
        on_mcp_context=None,
        on_turn_complete=None,
        on_frontend_inference=None,
        on_phrase_start=None,
        emit_tts=True,
        audio_in_flight_fn=None,
    ):
        self.convfill_config = convfill_config

        self.thought_queue = queue.Queue()
        self.done_event = threading.Event()
        self.cancel_event = threading.Event()
        self.tts_queue = queue.Queue()
        self.tts_model_path = convfill_config.tts_model_path if hasattr(convfill_config, "tts_model_path") else None

        # callbacks for web
        self.on_thought = on_thought
        self.on_response_fragment = on_response_fragment
        self.on_turn_complete = on_turn_complete
        self.on_frontend_inference = on_frontend_inference
        self.on_phrase_start = on_phrase_start
        self.emit_tts = emit_tts
        # count of utterances that have been emitted but not yet finished playing
        self.audio_in_flight_fn = (
            audio_in_flight_fn if audio_in_flight_fn is not None else (lambda: self.tts_queue.unfinished_tasks)
        )

        # bridges the gap between pulling a thought off thought_queue and the corresponding audio fragment showing up in audio_in_flight_fn() so backend_fn can't see all-zeros while frontend inference is in flight
        self._frontend_inflight = 0
        self._frontend_inflight_lock = threading.Lock()

        # Reasoner knowledge consumed by a turn that was cut short. Stashed by
        # run_frontend so commit_interrupted_turn can keep it.
        self._interrupted_thoughts = []

        # setup frontend, backend, dialogue state
        print("[LOG] Setting up Dialogue State Manager\n")
        if dialogue_state_manager is not None:
            self.dialogue_state_manager = dialogue_state_manager
        else:
            self.dialogue_state_manager = DialogueStateManager(num_history_turns=self.convfill_config.num_history_turns)

        print("[LOG] Setting up ConvFill Backend\n")
        self._turn_lock = threading.Lock()
        self._poisoned = False
        self._commit_lock = threading.RLock()
        self.api_key = ""
        self.convfill_backend = ConvFillBackend(
            dialogue_state_manager=self.dialogue_state_manager,
            prompt_template_path=self.convfill_config.backend_prompt_template_file,
            model_backend=BackendInference(
                api_key=self.api_key,
                model_name=self.convfill_config.backend_model_name,
                model_mode=self.convfill_config.backend_model_mode,
                remote_config=self.convfill_config.reasoner,
            ),
            mode=self.convfill_config.mode,
            task_specific_config=self.convfill_config.task_specific_config,
            on_rag_context=on_rag_context,
            on_mcp_context=on_mcp_context,
        )

        print("[LOG] Setting up ConvFill Frontend\n")
        self.convfill_frontend = ConvFillFrontend(
            model_config_path=self.convfill_config.frontend_model_config_path,
            dialogue_state_manager=self.dialogue_state_manager,
            device=self.convfill_config.frontend_device,
            dtype=self.convfill_config.frontend_dtype,
        )

    def backend_fn(self):
        q = queue.Queue()

        cancel = self.cancel_event
        frontend = getattr(self, "convfill_frontend", None)
        turn_state = getattr(frontend, "turn_state_manager", None)
        # A tool result resumes the same utterance. Give its knowledge stream
        # a short head start rather than immediately emitting a second preamble.
        filler_not_before = time.monotonic() + (1.0 if getattr(turn_state, "responses", []) else 0.0)

        def producer():
            gen = self.convfill_backend.backend_infer()
            try:
                for chunk in gen:
                    if cancel.is_set():
                        break
                    q.put(chunk)
            except BaseException as exc:
                if not cancel.is_set():
                    q.put(exc)
            finally:
                gen.close()
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while not self.cancel_event.is_set():
            try:
                chunk = q.get_nowait()
                if chunk is None:
                    return
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk
                continue
            except queue.Empty:
                pass

            # emit filler only when the listener would otherwise hear silence: no thought queued, no frontend inference in flight, and no audio currently playing.
            with self._frontend_inflight_lock:
                frontend_busy = self._frontend_inflight > 0
            if (
                time.monotonic() >= filler_not_before
                and self.thought_queue.qsize() == 0
                and not frontend_busy
                and self.audio_in_flight_fn() == 0
            ):
                yield "<sil>"
                continue

            try:
                chunk = q.get(timeout=0.1)
                if chunk is None:
                    return
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk
            except queue.Empty:
                continue

    def run_backend(self):
        try:
            for thought in self.backend_fn():
                if self.cancel_event.is_set():
                    break
                self.thought_queue.put(thought)
                if self.on_thought is not None:
                    self.on_thought(thought)
        finally:
            self.done_event.set()

    def run_frontend(self):
        t_prev_frontend_end = time.perf_counter()
        while not self.cancel_event.is_set() and (not self.done_event.is_set() or not self.thought_queue.empty()):
            try:
                thought = self.thought_queue.get(timeout=1)
                if thought == "<sil>":
                    with self.thought_queue.mutex:
                        has_real_pending = any(t != "<sil>" for t in self.thought_queue.queue)
                    if has_real_pending:
                        self.thought_queue.task_done()
                        continue

                phrase_start_ts = time.perf_counter()
                backend_gap_ms = (phrase_start_ts - t_prev_frontend_end) * 1000.0
                if self.on_phrase_start is not None:
                    self.on_phrase_start(thought, backend_gap_ms)
                with self._frontend_inflight_lock:
                    self._frontend_inflight += 1
                try:
                    for fragment in self.convfill_frontend.infer_and_update_state(thought):
                        if self.cancel_event.is_set():
                            break
                        if self.emit_tts:
                            cleaned = strip_markdown(fragment).strip()
                            if cleaned:
                                self.tts_queue.put(cleaned)
                        if self.on_response_fragment is not None:
                            self.on_response_fragment(fragment)
                    if self.on_frontend_inference is not None:
                        gen_ms = self.convfill_frontend.last_generate_ms
                        gen_tokens = self.convfill_frontend.last_generated_tokens
                        first_sentence_ms = self.convfill_frontend.last_first_sentence_ms
                        first_token_ms = self.convfill_frontend.last_first_token_ms
                        if gen_ms is not None:
                            self.on_frontend_inference(gen_ms, gen_tokens, first_sentence_ms, first_token_ms)
                finally:
                    with self._frontend_inflight_lock:
                        self._frontend_inflight -= 1
                    t_prev_frontend_end = time.perf_counter()
                self.thought_queue.task_done()
            except queue.Empty:
                continue
        if self.cancel_event.is_set():
            self._interrupted_thoughts = self.convfill_frontend.get_thoughts_and_reset_turn_state()
            return
        with self._commit_lock:
            if self.cancel_event.is_set():
                self._interrupted_thoughts = self.convfill_frontend.get_thoughts_and_reset_turn_state()
                return
            user_turn, final_response, final_thoughts = self.convfill_frontend.get_final_response_and_reset_turn_state()
            if self.on_turn_complete is not None:
                self.on_turn_complete(final_response, final_thoughts)

    def run_turn(self, user_input):
        """Join both stages, propagate failures, and quarantine stuck workers."""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("A ConvFill turn is already running")
        try:
            if self._poisoned:
                raise RuntimeError("Inference worker did not stop; restart this pipeline")
            self.dialogue_state_manager.update_user_state(user_input)
            self.done_event.clear()
            self.cancel_event = threading.Event()
            self._drain_turn_queues()
            for stage in (self.convfill_backend, self.convfill_frontend):
                begin = getattr(stage, "begin_turn", None)
                if begin:
                    begin(self.cancel_event)
            errors = queue.Queue()

            def run(fn):
                try:
                    fn()
                except BaseException as exc:
                    errors.put(exc)
                    self.cancel_turn()

            workers = [
                threading.Thread(target=run, args=(fn,), daemon=True) for fn in (self.run_backend, self.run_frontend)
            ]
            for worker in workers:
                worker.start()
            deadline = time.monotonic() + self.convfill_config.turn_timeout_s
            stopping_at = None
            while any(worker.is_alive() for worker in workers):
                if time.monotonic() >= deadline and not self.cancel_event.is_set():
                    errors.put(TimeoutError("ConvFill turn deadline exceeded"))
                    self.cancel_turn()
                if self.cancel_event.is_set():
                    stopping_at = stopping_at or time.monotonic()
                    if time.monotonic() - stopping_at >= self.convfill_config.shutdown_timeout_s:
                        self._poisoned = True
                        errors.put(TimeoutError("ConvFill inference worker did not stop"))
                        break
                for worker in workers:
                    worker.join(timeout=0.025)
            if not errors.empty():
                error = errors.get()
                self.dialogue_state_manager.rollback_turn(user_input)
                if not self._poisoned:
                    self.convfill_frontend.reset_turn_state()
                    self._drain_turn_queues()
                raise error
            if self.cancel_event.is_set():
                self._drain_turn_queues()
                return False
            self._interrupted_thoughts = []
            return True
        finally:
            self._turn_lock.release()

    def commit_interrupted_turn(self, user_input, heard_text):
        """Keep a best-effort dispatched prefix, marked as interrupted."""
        marked = f"{heard_text.strip()} [interrupted by the user]".strip()
        thoughts = list(self._interrupted_thoughts)
        self._interrupted_thoughts = []

        dsm = self.dialogue_state_manager
        if dsm.user_turns and dsm.user_turns[-1] == user_input and len(dsm.responses) == len(dsm.user_turns):
            # Cancellation raced the commit: replace the full response with the
            # heard prefix, keeping the thoughts that were already recorded.
            if not thoughts and dsm.thoughts:
                thoughts = list(dsm.thoughts[-1])
            dsm.rollback_turn(user_input)
            dsm.update_user_state(user_input)

        committed = dsm.commit_turn(user_input, marked, thoughts)
        self._drain_turn_queues()
        return committed

    def discard_pending_turn(self, user_input):
        """Drop a cancelled turn entirely; nothing of it was heard."""
        self._interrupted_thoughts = []
        removed = self.dialogue_state_manager.rollback_turn(user_input)
        self._drain_turn_queues()
        return removed

    def cancel_turn(self):
        """Cooperatively stop the active turn and prevent state commit."""
        self.cancel_event.set()
        self.done_event.set()
        for stage in (self.convfill_backend, getattr(self, "convfill_frontend", None)):
            cancel = getattr(stage, "cancel", None)
            if cancel:
                cancel()

    def set_audio_in_flight_fn(self, callback):
        """Replace the transport-specific audio activity probe."""
        self.audio_in_flight_fn = callback

    def _drain_turn_queues(self):
        for q in (self.thought_queue, self.tts_queue):
            try:
                while True:
                    q.get_nowait()
                    try:
                        q.task_done()
                    except ValueError:
                        pass
            except queue.Empty:
                pass

    def reset(self):
        """Clear all per-session dialogue and in-flight turn state."""
        self.cancel_turn()
        self.convfill_frontend.reset_turn_state()
        self._interrupted_thoughts = []
        self.dialogue_state_manager.reset()
        self._drain_turn_queues()
        self.done_event.clear()
