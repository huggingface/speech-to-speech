import base64
import io
import logging
import random
import re
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Any, Callable, Iterator, Optional, Sequence

import requests  # type: ignore[import-untyped]
from PIL import Image

logger = logging.getLogger(__name__)


SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

SPEECHABLE_PATTERN = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–\n\r\t]",
    flags=re.UNICODE,
)


def remove_unspeechable(text: str) -> str:
    """Keep only speechable characters: letters, digits, punctuation, whitespace.
    support unicode characters (english, arabic, chinese, japanese, korean, etc.)
    """
    text = text.translate(SMART_PUNCT_TRANSLATION)
    return SPEECHABLE_PATTERN.sub("", text)


# Maps an STT language code to the language name used in the "Please reply ... in {name}"
# prompt. Every language any bundled STT backend can report needs an entry here, otherwise
# `--enable_lang_prompt` silently emits no instruction for it. The names are lowercase
# because they are interpolated mid-sentence.
#
# `tests/test_llm_utils.py` asserts this covers the SUPPORTED_LANGUAGES of every bundled STT
# handler, so adding a language to a handler without adding it here fails CI.
WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
    # The remaining languages Parakeet TDT v3 (the default STT) detects and reports.
    "ru": "russian",
    "uk": "ukrainian",
    "cs": "czech",
    "sk": "slovak",
    "hu": "hungarian",
    "ro": "romanian",
    "bg": "bulgarian",
    "hr": "croatian",
    "sl": "slovenian",
    "sr": "serbian",
    "da": "danish",
    "no": "norwegian",
    "sv": "swedish",
    "fi": "finnish",
    "et": "estonian",
    "lv": "latvian",
    "lt": "lithuanian",
}


def resolve_auto_language(language_code: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Strip the ``-auto`` suffix and resolve the human-readable language name.

    Returns ``(clean_code, language_name)``.  ``language_name`` is non-None
    when the code (with or without ``-auto``) maps to a known language.
    """
    if not language_code:
        return language_code, None
    if language_code.endswith("-auto"):
        language_code = language_code[:-5]
    if language_code not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
        return language_code, None
    return language_code, WHISPER_LANGUAGE_TO_LLM_LANGUAGE.get(language_code)


def image_url_to_pil(image_url: str) -> Image.Image:
    """Convert an image URL or base64 data URI to a PIL Image.

    Accepts:
    - 'data:image/...;base64,<b64>' data URIs
    - 'https://...`` or ``http://...' URLs (fetched with a 10s timeout)
    """
    if image_url.startswith("data:"):
        _, b64_data = image_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64_data)))
    resp = requests.get(image_url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def run_generator_with_filler_sentences(
    gen_fn: Any,
    enable_filler_sentences: bool,
    filler_sentence_delay_s: float,
    filler_sentences: Sequence[str],
    language_code: Optional[str],
    runtime_config: Any,
    response: Any,
    turn_id: str | None,
    turn_revision: int | None,
    speech_stopped_at_s: float | None,
    gen: int | None,
    is_stale_fn: Optional[Any] = None,
) -> Iterator[Any]:
    """Wrap a generator function ``gen_fn`` to emit a filler sentence if no output is produced within ``filler_sentence_delay_s``.

    If ``enable_filler_sentences`` is False or ``filler_sentences`` is empty, yields directly from ``gen_fn()``.
    Otherwise, runs ``gen_fn()`` in a worker thread and monitors output latency. If the delay threshold is exceeded before any spoken content is yielded,
    a filler sentence chunk is emitted.
    """
    if not enable_filler_sentences or not filler_sentences:
        yield from gen_fn()
        return

    out_queue: Queue[Any] = Queue()
    _SENTINEL = object()

    def worker() -> None:
        try:
            for item in gen_fn():
                out_queue.put(item)
        except Exception as exc:
            out_queue.put(exc)
        finally:
            out_queue.put(_SENTINEL)

    worker_thread = Thread(target=worker, daemon=True)
    worker_thread.start()

    start_time = perf_counter()
    filler_emitted = False
    first_chunk_yielded = False

    while True:
        try:
            item = out_queue.get(timeout=0.05)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            from speech_to_speech.pipeline.messages import LLMResponseChunk

            if isinstance(item, LLMResponseChunk) and (item.text.strip() or item.tools):
                first_chunk_yielded = True

            yield item

        except Empty:
            if not first_chunk_yielded and not filler_emitted:
                elapsed = perf_counter() - start_time
                if elapsed >= filler_sentence_delay_s:
                    if is_stale_fn and is_stale_fn():
                        filler_emitted = True
                        continue

                    filler_text = random.choice(list(filler_sentences))
                    logger.info(
                        "LLM response latency (%.2fs) exceeded threshold (%.2fs); emitting filler sentence: '%s'",
                        elapsed,
                        filler_sentence_delay_s,
                        filler_text,
                    )
                    from speech_to_speech.pipeline.messages import LLMResponseChunk

                    yield LLMResponseChunk(
                        text=filler_text,
                        language_code=language_code,
                        runtime_config=runtime_config,
                        response=response,
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                        speech_stopped_at_s=speech_stopped_at_s,
                        cancel_generation=gen,
                    )
                    filler_emitted = True

    worker_thread.join(timeout=1.0)

