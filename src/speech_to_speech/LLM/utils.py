import base64
import io
import re
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

# A word after an opening fence is a language tag only when the fence line ends.
# Requiring the newline preserves same-line code spans such as ```code```.
MARKDOWN_FENCE_OPEN_PATTERN = re.compile(
    r"^[ \t]{0,3}`{3,}[ \t]*[\w+#-]*[ \t]*\r?\n",
    flags=re.MULTILINE,
)
MARKDOWN_FENCE_CLOSE_PATTERN = re.compile(r"^[ \t]{0,3}`{3,}[ \t]*$", flags=re.MULTILINE)

# Matched star pairs may directly adjoin CJK or other word characters. The
# fallback delimiter pass still requires boundaries, preserving unmatched
# compact operators such as `2*3` and `5**2`.
MARKDOWN_DOUBLE_STAR_PAIR_PATTERN = re.compile(r"(?<!\*)\*\*(?!\*)(?P<body>[^\s*](?:[^\n]*?[^\s*])?)\*\*(?!\*)")
MARKDOWN_SINGLE_STAR_PAIR_PATTERN = re.compile(r"(?<!\*)\*(?!\*)(?P<body>[^\s*](?:[^\n]*?[^\s*])?)\*(?!\*)")
MARKDOWN_STAR_DELIMITER_PATTERN = re.compile(r"(?<![\w*])\*{1,3}(?=\S)|(?<=\S)\*{1,3}(?![\w*])")
MARKDOWN_UNDERSCORE_DELIMITER_PATTERN = re.compile(r"(?<!\w)_{1,2}(?=\S)|(?<=\S)_{1,2}(?!\w)")
MARKDOWN_BACKTICK_DELIMITER_PATTERN = re.compile(r"(?<=\S)`{1,3}|`{1,3}(?=\S)")


def remove_markdown(text: str) -> str:
    """Strip common Markdown delimiters while preserving the enclosed text.

    Must run on complete text, not per-token deltas: a delimiter run can arrive
    split across two streaming chunks.
    """
    protected_code: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        token = f"\x00markdown-code-{len(protected_code)}\x00"
        protected_code.append(match.group("body"))
        return token

    text = MARKDOWN_FENCED_CODE_PATTERN.sub(protect_code, text)
    text = MARKDOWN_INLINE_CODE_PATTERN.sub(protect_code, text)
    text = MARKDOWN_HEADING_PATTERN.sub("", text)
    text = MARKDOWN_BULLET_PATTERN.sub("", text)
    text = MARKDOWN_FENCE_OPEN_PATTERN.sub("", text)
    text = MARKDOWN_FENCE_CLOSE_PATTERN.sub("", text)
    text = MARKDOWN_DOUBLE_STAR_PAIR_PATTERN.sub(r"\g<body>", text)
    text = MARKDOWN_SINGLE_STAR_PAIR_PATTERN.sub(r"\g<body>", text)
    text = MARKDOWN_STAR_DELIMITER_PATTERN.sub("", text)
    text = MARKDOWN_UNDERSCORE_DELIMITER_PATTERN.sub("", text)
    text = MARKDOWN_BACKTICK_DELIMITER_PATTERN.sub("", text)
    for index, code_body in enumerate(protected_code):
        text = text.replace(f"\x00markdown-code-{index}\x00", code_body)
    return text


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
