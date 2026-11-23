"""Grapheme-to-phoneme frontend for Pocket TTS Farsi v2."""

from speech_to_speech.TTS.persian_normalization import normalize

G2P_MODEL = "mehdi-hf/Homo-GE2PE-Persian-HF"
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})


class PersianPhonemizer:
    def __init__(self, device: str = "cpu") -> None:
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(G2P_MODEL)
        # Transformers wraps from_pretrained with a decorator that loses its classmethod type.
        self.model = T5ForConditionalGeneration.from_pretrained(G2P_MODEL).to(device).eval()  # type: ignore[arg-type]

    def __call__(self, text: str) -> str:
        import torch

        text = normalize(text).replace("؟", "").replace("?", "")
        if not text.strip():
            return ""
        encoded = self.tokenizer([text], add_special_tokens=False, return_tensors="pt").to(self.device)
        # Refuse truncation: dropping the rest of a sentence would lose speech.
        if encoded["input_ids"].shape[-1] > 512:
            raise ValueError(
                "Pocket TTS Farsi sentence exceeds the G2P input limit of 512 tokens. Split it into shorter sentences."
            )
        with torch.inference_mode():
            output = self.model.generate(**encoded, num_beams=5, max_length=512, early_stopping=True)
        if output.shape[-1] >= 512:
            raise ValueError(
                "Pocket TTS Farsi phoneme conversion hit its length limit. Split the text into shorter sentences."
            )
        raw = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        return raw.translate(TO_PHONEMES)
