import logging
import os
import socket
import sys
import threading
from copy import copy
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import perf_counter
from typing import Optional
from sys import platform
from arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from arguments_classes.mlx_language_model_arguments import MLXLanguageModelHandlerArguments
from arguments_classes.module_arguments import ModuleArguments
from arguments_classes.parler_tts_arguments import ParlerTTSHandlerArguments
from arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from arguments_classes.socket_sender_arguments import SocketSenderArguments
from arguments_classes.vad_arguments import VADHandlerArguments
from arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from baseHandler import BaseHandler
from arguments_classes.melo_tts_arguments import MeloTTSHandlerArguments
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    TextIteratorStreamer,
)
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
import librosa

from local_audio_streamer import LocalAudioStreamer
from utils import VADIterator, int2float, next_power_of_2

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")


console = Console()


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for handler in self.handlers:
            handler.stop_event.set()
        for thread in self.threads:
            thread.join()


class SocketReceiver:
    """
    Handles reception of the audio packets from the client.
    """

    def __init__(
        self,
        stop_event,
        queue_out,
        should_listen,
        host="0.0.0.0",
        port=12345,
        chunk_size=1024,
    ):
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.chunk_size = chunk_size
        self.host = host
        self.port = port

    def receive_full_chunk(self, conn, chunk_size):
        data = b""
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                # connection closed
                return None
            data += packet
        return data

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Receiver waiting to be connected...")
        self.conn, _ = self.socket.accept()
        logger.info("receiver connected")

        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                # connection closed
                self.queue_out.put(b"END")
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        self.conn.close()
        logger.info("Receiver closed")


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12346):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Sender waiting to be connected...")
        self.conn, _ = self.socket.accept()
        logger.info("sender connected")

        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            self.conn.sendall(audio_chunk)
            if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                break
        self.conn.close()
        logger.info("Sender closed")


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"audio input of duration: {len(array) / self.sample_rate}s, skipping"
                )
            else:
                self.should_listen.clear()
                logger.debug("Stop listening")
                yield array


class WhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name="distil-whisper/distil-large-v3",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.gen_kwargs = gen_kwargs

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)

        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=self.compile_mode, fullgraph=True
            )
        self.warmup()

    def prepare_model_inputs(self, spoken_prompt):
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1, self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device,
        )
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs,
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        input_features = self.prepare_model_inputs(spoken_prompt)
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)
        pred_text = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True, decode_with_timestamps=False
        )[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message):
        self.init_chat_message = init_chat_message

    def to_list(self):
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer


class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        device="cuda",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=device
        )
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs,
        }

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Write me a poem about Machine Learning."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": self.gen_kwargs["max_new_tokens"],
            "max_new_tokens": self.gen_kwargs["max_new_tokens"],
            **self.gen_kwargs,
        }

        n_steps = 2

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            thread = Thread(
                target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs
            )
            thread.start()
            for _ in self.streamer:
                pass

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, prompt):
        logger.debug("infering language model...")

        self.chat.append({"role": self.user_role, "content": prompt})
        thread = Thread(
            target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs
        )
        thread.start()
        if self.device == "mps":
            generated_text = ""
            for new_text in self.streamer:
                generated_text += new_text
            printable_text = generated_text
            torch.mps.empty_cache()
        else:
            generated_text, printable_text = "", ""
            for new_text in self.streamer:
                generated_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    yield (sentences[0])
                    printable_text = new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        # don't forget last sentence
        yield printable_text


class ParlerTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        model_name="ylacombe/parler-tts-mini-jenny-30H",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
        max_prompt_pad_length=8,
        description=(
            "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
            "She speaks very fast."
        ),
        play_steps_s=1,
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs
        self.compile_mode = compile_mode
        self.max_prompt_pad_length = max_prompt_pad_length
        self.description = description

        self.description_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        ).to(device)

        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)
        self.blocksize = blocksize

        if self.compile_mode not in (None, "default"):
            logger.warning(
                "Torch compilation modes that captures CUDA graphs are not yet compatible with the STT part. Reverting to 'default'"
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

        tokenized_description = self.description_tokenizer(
            self.description, return_tensors="pt"
        )
        input_ids = tokenized_description.input_ids.to(self.device)
        attention_mask = tokenized_description.attention_mask.to(self.device)

        tokenized_prompt = self.prompt_tokenizer(
            prompt, return_tensors="pt", **pad_args_prompt
        )
        prompt_input_ids = tokenized_prompt.input_ids.to(self.device)
        prompt_attention_mask = tokenized_prompt.attention_mask.to(self.device)

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


def prepare_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """

    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def main():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            LanguageModelHandlerArguments,
            MLXLanguageModelHandlerArguments,
            ParlerTTSHandlerArguments,
            MeloTTSHandlerArguments,
        )
    )

    # 0. Parse CLI arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse configurations from a JSON file if specified
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            mlx_language_model_handler_kwargs,
            parler_tts_handler_kwargs,
            melo_tts_handler_kwargs,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse arguments from command line if no JSON file is provided
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            mlx_language_model_handler_kwargs,
            parler_tts_handler_kwargs,
            melo_tts_handler_kwargs,
        ) = parser.parse_args_into_dataclasses()

    # 1. Handle logger
    global logger
    logging.basicConfig(
        level=module_kwargs.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if module_kwargs.log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

    def optimal_mac_settings(mac_optimal_settings: Optional[str], *handler_kwargs):
        if mac_optimal_settings:
            for kwargs in handler_kwargs:
                if hasattr(kwargs, "device"):
                    kwargs.device = "mps"
                if hasattr(kwargs, "mode"):
                    kwargs.mode = "local"
                if hasattr(kwargs, "stt"):
                    kwargs.stt = "whisper-mlx"
                if hasattr(kwargs, "llm"):
                    kwargs.llm = "mlx-lm"
                if hasattr(kwargs, "tts"):
                    kwargs.tts = "melo"

    optimal_mac_settings(
        module_kwargs.local_mac_optimal_settings,
        module_kwargs,
    )

    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError(
                "Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'."
            )
        if module_kwargs.llm != "mlx-lm":
            logger.warning(
                "For macOS users, it is recommended to use mlx-lm. You can activate it by passing --llm mlx-lm."
            )
        if module_kwargs.tts != "melo":
            logger.warning(
                "If you experiences issues generating the voice, considering setting the tts to melo."
            )

    # 2. Prepare each part's arguments
    def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
        if common_device:
            for kwargs in handler_kwargs:
                if hasattr(kwargs, "lm_device"):
                    kwargs.lm_device = common_device
                if hasattr(kwargs, "tts_device"):
                    kwargs.tts_device = common_device
                if hasattr(kwargs, "stt_device"):
                    kwargs.stt_device = common_device

    # Call this function with the common device and all the handlers
    overwrite_device_argument(
        module_kwargs.device,
        language_model_handler_kwargs,
        mlx_language_model_handler_kwargs,
        parler_tts_handler_kwargs,
        whisper_stt_handler_kwargs,
    )

    prepare_args(whisper_stt_handler_kwargs, "stt")
    prepare_args(language_model_handler_kwargs, "lm")
    prepare_args(mlx_language_model_handler_kwargs, "mlx_lm")
    prepare_args(parler_tts_handler_kwargs, "tts")
    prepare_args(melo_tts_handler_kwargs, "melo")

    # 3. Build the pipeline
    stop_event = Event()
    # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
    should_listen = Event()
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue()
    text_prompt_queue = Queue()
    lm_response_queue = Queue()

    if module_kwargs.mode == "local":
        local_audio_streamer = LocalAudioStreamer(
            input_queue=recv_audio_chunks_queue, output_queue=send_audio_chunks_queue
        )
        comms_handlers = [local_audio_streamer]
        should_listen.set()
    else:
        comms_handlers = [
            SocketReceiver(
                stop_event,
                recv_audio_chunks_queue,
                should_listen,
                host=socket_receiver_kwargs.recv_host,
                port=socket_receiver_kwargs.recv_port,
                chunk_size=socket_receiver_kwargs.chunk_size,
            ),
            SocketSender(
                stop_event,
                send_audio_chunks_queue,
                host=socket_sender_kwargs.send_host,
                port=socket_sender_kwargs.send_port,
            ),
        ]

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )
    if module_kwargs.stt == "whisper":
        stt = WhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "whisper-mlx":
        from STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler
        stt = LightningWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    else:
        raise ValueError("The STT should be either whisper or whisper-mlx")
    if module_kwargs.llm == "transformers":
        lm = LanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(language_model_handler_kwargs),
        )
    elif module_kwargs.llm == "mlx-lm":
        from LLM.mlx_lm import MLXLanguageModelHandler
        lm = MLXLanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(mlx_language_model_handler_kwargs),
        )
    else:
        raise ValueError("The LLM should be either transformers or mlx-lm")
    if module_kwargs.tts == "parler":
        torch._inductor.config.fx_graph_cache = True
        # mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
        torch._dynamo.config.cache_size_limit = 15
        tts = ParlerTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(parler_tts_handler_kwargs),
        )

    elif module_kwargs.tts == "melo":
        try:
            from TTS.melotts import MeloTTSHandler
        except RuntimeError as e:
            logger.error(
                "Error importing MeloTTSHandler. You might need to run: python -m unidic download"
            )
            raise e
        tts = MeloTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(melo_tts_handler_kwargs),
        )
    else:
        raise ValueError("The TTS should be either parler or melo")

    # 4. Run the pipeline
    try:
        pipeline_manager = ThreadManager([*comms_handlers, vad, stt, lm, tts])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    main()
