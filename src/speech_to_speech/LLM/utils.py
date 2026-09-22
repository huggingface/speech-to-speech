import base64
import io
import re
from collections.abc import Callable
from typing import Optional

import requests  # type: ignore[import-untyped]
from PIL import Image

SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

SPEECHABLE_PATTERN = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–，。！？；：、\n\r\t]",
    flags=re.UNICODE,
)

MARKDOWN_HEADING_PATTERN = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+|$)", flags=re.MULTILINE)
MARKDOWN_BULLET_PATTERN = re.compile(r"^[ \t]{0,3}[-*+][ \t]+", flags=re.MULTILINE)

# Protect complete code bodies while prose-only Markdown passes run. Fenced
# blocks are handled first so their backticks are not mistaken for inline code.
MARKDOWN_FENCED_CODE_PATTERN = re.compile(
    r"^[ \t]{0,3}`{3,}[^\r\n]*\r?\n(?P<body>.*?)(?:^[ \t]{0,3}`{3,}[ \t]*$)",
    flags=re.MULTILINE | re.DOTALL,
)
MARKDOWN_INLINE_CODE_PATTERN = re.compile(r"(?P<ticks>`{1,})(?P<body>[^\n]*?)(?P=ticks)")
MARKDOWN_FENCE_LINE_PATTERN = re.compile(
    r"^[ \t]{0,3}(?P<ticks>`{3,})(?P<rest>[^\r\n]*)$",
    flags=re.MULTILINE,
)

# Emphasis is removed only when the same delimiter run opens and closes it at
# conservative word boundaries. Intraword stars are preserved as operators.
MARKDOWN_BOUNDARY_EMPHASIS_PATTERN = re.compile(
    r"(?<![\w*_])(?P<delimiter>\*{1,3}|_{1,2})(?![*_])"
    r"(?P<body>\S(?:[^\n]*?\S)?)(?P=delimiter)(?![\w*_])"
)


def _protect_markdown_code(text: str, *, keep_delimiters: bool) -> tuple[str, list[str]]:
    protected_code: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        token = f"\x00markdown-code-{len(protected_code)}\x00"
        protected_code.append(match.group(0) if keep_delimiters else match.group("body"))
        return token

    text = MARKDOWN_FENCED_CODE_PATTERN.sub(protect_code, text)
    text = MARKDOWN_INLINE_CODE_PATTERN.sub(protect_code, text)
    return text, protected_code


def _restore_markdown_code(text: str, protected_code: list[str]) -> str:
    for index, code_body in enumerate(protected_code):
        text = text.replace(f"\x00markdown-code-{index}\x00", code_body)
    return text


def _protect_matched_emphasis(text: str) -> tuple[str, list[str]]:
    protected_emphasis: list[str] = []

    def protect_emphasis(match: re.Match[str]) -> str:
        token = f"\x00markdown-emphasis-{len(protected_emphasis)}\x00"
        protected_emphasis.append(match.group(0))
        return token

    text = MARKDOWN_BOUNDARY_EMPHASIS_PATTERN.sub(protect_emphasis, text)
    return text, protected_emphasis


def _restore_matched_emphasis(text: str, protected_emphasis: list[str]) -> str:
    for index, emphasis in enumerate(protected_emphasis):
        text = text.replace(f"\x00markdown-emphasis-{index}\x00", emphasis)
    return text


def _has_unclosed_markdown_fence(text: str) -> bool:
    opening_ticks: int | None = None
    for match in MARKDOWN_FENCE_LINE_PATTERN.finditer(text):
        ticks = len(match.group("ticks"))
        rest = match.group("rest")
        if opening_ticks is None:
            # Backticks later on the same line make this an inline code span,
            # such as the issue's ```code``` example, rather than a fence.
            if "`" not in rest:
                opening_ticks = ticks
        elif ticks >= opening_ticks and not rest.strip():
            opening_ticks = None
    return opening_ticks is not None


def sent_tokenize_preserving_markdown_code(
    text: str,
    tokenizer: Callable[[str], list[str]],
) -> list[str]:
    """Tokenize prose without splitting complete Markdown constructs."""
    if _has_unclosed_markdown_fence(text):
        return [text]
    protected_text, protected_code = _protect_markdown_code(text, keep_delimiters=True)
    protected_text, protected_emphasis = _protect_matched_emphasis(protected_text)
    return [
        _restore_markdown_code(_restore_matched_emphasis(sentence, protected_emphasis), protected_code)
        for sentence in tokenizer(protected_text)
    ]


def remove_markdown(text: str) -> str:
    """Strip common Markdown delimiters while preserving the enclosed text.

    Must run on complete text, not per-token deltas: a delimiter run can arrive
    split across two streaming chunks.
    """
    text, protected_code = _protect_markdown_code(text, keep_delimiters=False)
    text = MARKDOWN_HEADING_PATTERN.sub("", text)
    text = MARKDOWN_BULLET_PATTERN.sub("", text)
    text = MARKDOWN_BOUNDARY_EMPHASIS_PATTERN.sub(r"\g<body>", text)
    return _restore_markdown_code(text, protected_code)


def remove_unspeechable(text: str) -> str:
    """Keep only speechable characters: letters, digits, punctuation, whitespace.
    support unicode characters (english, arabic, chinese, japanese, korean, etc.)

    Safe to call per streaming delta. Markdown stripping is intentionally not
    included here -- unlike character filtering, it needs complete text (see
    remove_markdown), so callers apply it separately once a full sentence exists.
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
    # The rest of Whisper's language set, which the Whisper-family backends (including
    # faster-whisper) can detect and report. Without a name here the prompt is silently
    # skipped for that language, so coverage has to match what the backends can emit.
    "tr": "turkish",
    "ca": "catalan",
    "ar": "arabic",
    "id": "indonesian",
    "vi": "vietnamese",
    "he": "hebrew",
    "el": "greek",
    "ms": "malay",
    "ta": "tamil",
    "th": "thai",
    "ur": "urdu",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "te": "telugu",
    "fa": "persian",
    "bn": "bengali",
    "az": "azerbaijani",
    "kn": "kannada",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    # Qwen3-ASR reports Filipino as "fil"; Whisper only knows "tl" (Tagalog) above.
    "fil": "filipino",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}


def format_language_instruction(lang_name: str) -> str:
    """Build the opt-in language instruction for ``--enable_lang_prompt``."""
    return f"Please reply to my message in {lang_name}."


def language_name_for_prompt(language_code: Optional[str], *, enable: bool) -> Optional[str]:
    """Resolve the language name injected into the system prompt when enabled."""
    if not enable:
        return None
    _, lang_name = resolve_auto_language(language_code)
    return lang_name


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
