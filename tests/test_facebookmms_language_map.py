"""Facebook MMS must key its model map by Whisper language codes.

STT backends emit Whisper-style codes (e.g. ``pl``). Wrong keys silently fall
back to English in ``FacebookMMSTTSHandler.load_model``.
"""

from speech_to_speech.TTS.facebookmms_handler import WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE

# Whisper-style ISO 639-1 codes expected as keys of this map. Keep the set
# explicit so a typo like "po" instead of "pl" fails CI.
_KNOWN_WHISPER_CODES = {
    "en",
    "fr",
    "es",
    "ko",
    "hi",
    "ar",
    "hy",
    "az",
    "bg",
    "ca",
    "nl",
    "fi",
    "de",
    "el",
    "he",
    "hu",
    "is",
    "id",
    "kn",
    "kk",
    "lv",
    "ms",
    "mr",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
}


def test_facebookmms_map_keys_are_whisper_codes() -> None:
    unknown = sorted(set(WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE) - _KNOWN_WHISPER_CODES)
    assert unknown == [], f"Non-Whisper keys in MMS map (would never match STT): {unknown}"


def test_common_stt_languages_resolve_to_mms_models() -> None:
    # Failure mode: STT emits "pl" / "bg" / … and the map must hit the MMS model.
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["pl"] == "pol"
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["bg"] == "bul"
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["tr"] == "tur"
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["tl"] == "tgl"
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["mr"] == "mar"
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["ms"] == "zlm"
    assert WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE["kn"] == "kan"
