"""Conversation-side TTS lifecycle, independent of the allocated GPU provider."""

import logging
from threading import Event
from time import perf_counter

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.turn_latency import record_response_e2e, record_response_tts_ttfa
from speech_to_speech.zerogpu import runtime

logger = logging.getLogger(__name__)


class ZeroGPUTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen,
        backend=None,
        model_name=None,
        voice=None,
        language=None,
        gen_kwargs=None,
        cancel_scope=None,
        speculative_turns=None,
        detect_llm_output_language=False,
    ):
        config = runtime.get_tts_config()
        if backend is not None and backend != config["backend"]:
            raise ValueError("ZeroGPU TTS backend must match the initialized runtime")
        self.detect_llm_output_language = detect_llm_output_language
        if model_name is not None and model_name != config["model"]:
            raise ValueError("ZeroGPU TTS model must match the initialized runtime")
        if gen_kwargs:
            raise ValueError("Configure ZeroGPU TTS generation options on the runtime provider")
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.voice = runtime.resolve_tts_voice(voice, config["voice"])
        self._initial_voice = self.voice
        self.language = language or config["language"]
        self.supported_languages = config.get("supported_languages")
        self._failed_responses = set()
        self._session_stopped = Event()

    def _identity(self, item):
        generation = item.cancel_generation
        if generation is None and self.cancel_scope is not None:
            generation = self.cancel_scope.generation
        return generation, item.response_key, item.turn_id, item.turn_revision

    def _cancelled(self, item, generation):
        return (
            self.stop_event.is_set()
            or (self.cancel_scope is not None and self.cancel_scope.is_stale(generation))
            or (
                self.speculative_turns is not None
                and not self.speculative_turns.is_latest(item.turn_id, item.turn_revision)
            )
        )

    def _request_voice(self, item):
        requested = None
        if item.response and item.response.audio and item.response.audio.output:
            requested = item.response.audio.output.voice
        if not requested and item.runtime_config is not None:
            audio = item.runtime_config.session.audio
            if audio is not None and audio.output is not None:
                requested = audio.output.voice
        self.voice = runtime.resolve_tts_voice(requested, self.voice)
        return self.voice

    def _request_language(self, item):
        selected = item.selected_language
        if selected == "auto":
            selected = None
        language = item.tts_language_code or selected or self.language
        if self.supported_languages and language not in self.supported_languages:
            language = selected if selected in self.supported_languages else self.language
        return language

    def process(self, item):
        identity = self._identity(item)
        tracker = self.speculative_turns
        if isinstance(item, EndOfResponse):
            self._failed_responses.discard(identity)
            if tracker and not tracker.wait_for_gate(item.turn_id, item.turn_revision):
                if item.response_key is None:
                    return
                item.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return
        generation = identity[0]
        if identity in self._failed_responses or not item.text.strip() or self._cancelled(item, generation):
            return
        if tracker and not tracker.wait_for_gate(item.turn_id, item.turn_revision):
            return
        session_stopped = self._session_stopped

        def cancelled():
            return session_stopped.is_set() or self._cancelled(item, generation)

        chunks = None
        started = perf_counter()
        first = True
        try:
            chunks = runtime.synthesize_chunks(
                item.text, self._request_language(item), self._request_voice(item), cancelled=cancelled
            )
            for chunk in chunks:
                if cancelled():
                    return
                if len(chunk) == 0:
                    continue
                if first:
                    if tracker and not tracker.wait_for_gate(item.turn_id, item.turn_revision, commit=True):
                        return
                    store = getattr(self, "turn_latency_store", None)
                    record_response_tts_ttfa(store, item.response_key, perf_counter() - started)
                    if item.speech_stopped_at_s is not None:
                        record_response_e2e(store, item.response_key, max(0, perf_counter() - item.speech_stopped_at_s))
                    first = False
                yield chunk
            if first and not cancelled():
                raise RuntimeError("ZeroGPU TTS provider returned no audio")
        except Exception:
            logger.exception("ZeroGPU TTS synthesis failed")
            if not cancelled():
                self._failed_responses.add(identity)
                self.queue_out.put(
                    ResponseFailedEvent(
                        message="ZeroGPU TTS synthesis failed",
                        turn_id=item.turn_id,
                        turn_revision=item.turn_revision,
                        cancel_generation=generation,
                        response_key=item.response_key,
                    )
                )
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

    def on_session_end(self):
        self._session_stopped.set()
        self._session_stopped = Event()
        self.voice = self._initial_voice
        self._failed_responses.clear()

    def cleanup(self):
        # BaseHandler also calls cleanup when setup failed before state existed.
        if hasattr(self, "_session_stopped"):
            self.on_session_end()
