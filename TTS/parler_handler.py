from threading import Thread
from time import perf_counter
from baseHandler import BaseHandler
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
)
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
import librosa
import logging
from rich.console import Console
from utils.utils import next_power_of_2
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
)

torch._inductor.config.fx_graph_cache = True
# mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
torch._dynamo.config.cache_size_limit = 15

logger = logging.getLogger(__name__)

console = Console()


if not is_flash_attn_2_available() and torch.cuda.is_available():
    logger.warn(
        """Parler TTS works best with flash attention 2, but is not installed
        Given that CUDA is available in this system, you can install flash attention 2 with `uv pip install flash-attn --no-build-isolation`"""
    )


WHISPER_LANGUAGE_TO_PARLER_SPEAKER = {
    "en": "Jason",
    "fr": "Christine",
    "es": "Steven",
    "de": "Nicole",
    "pt": "Sophia",
    "pl": "Alex",
    "it": "Richard",
    "nl": "Mark",
}


class ParlerTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        model_name="parler-tts/parler-mini-v1-jenny",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
        max_prompt_pad_length=8,
        description=(
            "Jenny speaks at a slightly slow pace with an animated delivery with clear audio quality."
        ),
        play_steps_s=1,
        blocksize=512,
        use_default_speakers_list=True,
    ):
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs
        self.compile_mode = compile_mode
        self.max_prompt_pad_length = max_prompt_pad_length
        self.use_default_speakers_list = use_default_speakers_list
        if self.use_default_speakers_list:
            description = description.replace("Jenny", "")

        self.speaker = "Jason"
        self.description = description

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        ).to(device)
        
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)


        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)
        self.blocksize = blocksize

        if self.compile_mode not in (None, "default"):
            logger.warning(
                "Torch compilation modes that captures CUDA graphs are not yet compatible with the TTS part. Reverting to 'default'"
            )
            self.compile_mode = "default"

        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=self.compile_mode, fullgraph=True
            )

        self.warmup()

    def prepare_model_inputs(
        self,
        prompt,
        max_length_prompt=50,
        pad=False,
    ):
        pad_args_prompt = (
            {"padding": "max_length", "max_length": max_length_prompt} if pad else {}
        )

        description = self.description
        if self.use_default_speakers_list:
            description = self.speaker + " " + self.description

        tokenized_description = self.description_tokenizer(
            description, return_tensors="pt"
        ).to(self.device)
        input_ids = tokenized_description.input_ids
        attention_mask = tokenized_description.attention_mask

        tokenized_prompt = self.prompt_tokenizer(
            prompt, return_tensors="pt", **pad_args_prompt
        ).to(self.device)
        prompt_input_ids = tokenized_prompt.input_ids
        prompt_attention_mask = tokenized_prompt.attention_mask

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            **self.gen_kwargs,
        }

        return gen_kwargs

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1 if self.compile_mode == "default" else 2

        if self.device == "cuda":
            torch.cuda.synchronize()
            start_event.record()
        if self.compile_mode:
            pad_lengths = [2**i for i in range(2, self.max_prompt_pad_length)]
            for pad_length in pad_lengths[::-1]:
                model_kwargs = self.prepare_model_inputs(
                    "dummy prompt", max_length_prompt=pad_length, pad=True
                )
                for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                logger.info(f"Warmed up length {pad_length} tokens!")
        else:
            model_kwargs = self.prepare_model_inputs("dummy prompt")
            for _ in range(n_steps):
                _ = self.model.generate(**model_kwargs)

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            self.speaker = WHISPER_LANGUAGE_TO_PARLER_SPEAKER.get(language_code, "Jason")
            
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        nb_tokens = len(self.prompt_tokenizer(llm_sentence).input_ids)

        pad_args = {}
        if self.compile_mode:
            # pad to closest upper power of two
            pad_length = next_power_of_2(nb_tokens)
            logger.debug(f"padding to {pad_length}")
            pad_args["pad"] = True
            pad_args["max_length_prompt"] = pad_length

        tts_gen_kwargs = self.prepare_model_inputs(
            llm_sentence,
            **pad_args,
        )

        streamer = ParlerTTSStreamer(
            self.model, device=self.device, play_steps=self.play_steps
        )
        tts_gen_kwargs = {"streamer": streamer, **tts_gen_kwargs}
        torch.manual_seed(0)
        thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
        thread.start()

        for i, audio_chunk in enumerate(streamer):
            global pipeline_start
            if i == 0 and "pipeline_start" in globals():
                logger.info(
                    f"Time to first audio: {perf_counter() - pipeline_start:.3f}"
                )
            audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            for i in range(0, len(audio_chunk), self.blocksize):
                yield np.pad(
                    audio_chunk[i : i + self.blocksize],
                    (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
                )

        self.should_listen.set()
