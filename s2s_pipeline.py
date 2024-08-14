import logging
import os
import socket
import sys
import threading
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import perf_counter

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
    TextIteratorStreamer
)

from parler_tts import (
    ParlerTTSForConditionalGeneration,
    ParlerTTSStreamer
)

from utils import (
    VADIterator,
    int2float,
    next_power_of_2
)

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp") 
torch._inductor.config.fx_graph_cache = True
# mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
torch._dynamo.config.cache_size_limit = 15


console = Console()


@dataclass
class ModuleArguments:
    log_level: str = field(
        default="info",
        metadata={
            "help": "Provide logging level. Example --log_level debug, default=warning."
        }
    )


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


class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []

    def setup(self):
        pass

    def process(self):
        raise NotImplementedError

    def run(self):
        while not self.stop_event.is_set():
            input = self.queue_in.get()
            if isinstance(input, bytes) and input == b'END':
                # sentinelle signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            start_time = perf_counter()
            for output in self.process(input):
                self._times.append(perf_counter() - start_time)
                logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
                self.queue_out.put(output)
                start_time = perf_counter()

        self.cleanup()
        self.queue_out.put(b'END')

    @property
    def last_time(self):
        return self._times[-1]

    def cleanup(self):
        pass


@dataclass
class SocketReceiverArguments:
    recv_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP ddress for the socket connection. Default is '0.0.0.0' which binds to all "
                    "available interfaces on the host machine."
        }
    )
    recv_port: int = field(
        default=12345,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        }
    )
    chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of each data chunk to be sent or received over the socket. Default is 1024 bytes."
        }
    )


class SocketReceiver:
    """
    Handles reception of the audio packets from the client.
    """

    def __init__(
        self, 
        stop_event,
        queue_out,
        should_listen,
        host='0.0.0.0', 
        port=12345,
        chunk_size=1024
    ):  
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.chunk_size=chunk_size
        self.host = host
        self.port = port

    def receive_full_chunk(self, conn, chunk_size):
        data = b''
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
        logger.info('Receiver waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("receiver connected")

        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                # connection closed
                self.queue_out.put(b'END')
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        self.conn.close()
        logger.info("Receiver closed")


@dataclass
class SocketSenderArguments:
    send_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all "
                    "available interfaces on the host machine."
        }
    )
    send_port: int = field(
        default=12346,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        }
    )

            
class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(
        self, 
        stop_event,
        queue_in,
        host='0.0.0.0', 
        port=12346
    ):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port
        

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Sender waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("sender connected")

        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            self.conn.sendall(audio_chunk)
            if isinstance(audio_chunk, bytes) and audio_chunk == b'END':
                break
        self.conn.close()
        logger.info("Sender closed")


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.3,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        }
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        }
    )
    min_silence_ms: int = field(
        default=250,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 1000 ms."
        }
    )
    min_speech_ms: int = field(
        default=750,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
        }
    )
    max_speech_ms: float = field(
        default=float('inf'),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        }
    )
    speech_pad_ms: int = field(
        default=30,
        metadata={
            "help": "Amount of padding added to the beginning and end of detected speech segments. Measured in milliseconds. Default is 30 ms."
        }
    )


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
            max_speech_ms=float('inf'),
            speech_pad_ms=30,

        ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
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
                logger.debug(f"audio input of duration: {len(array) / self.sample_rate}s, skipping")
            else:
                self.should_listen.clear()
                logger.debug("Stop listening")
                yield array


@dataclass
class WhisperSTTHandlerArguments:
    stt_model_name: str = field(
        default="distil-whisper/distil-large-v3",
        metadata={
            "help": "The pretrained Whisper model to use. Default is 'distil-whisper/distil-large-v3'."
        }
    )
    stt_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        } 
    )
    stt_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        }
    )
    stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        }
    )
    stt_gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        }
    )
    stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        }
    )
    stt_gen_task: str = field(
        default="transcribe",
        metadata={
            "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
        }
    )
    stt_gen_language: str = field(
        default="en",
        metadata={
            "help": "The language of the speech to transcribe. Default is 'en' for English."
        }
    )


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
            gen_kwargs={}
        ): 
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode=compile_mode
        self.gen_kwargs = gen_kwargs

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)
        
        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
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
            (1,  self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device
        ) 
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        input_features = self.prepare_model_inputs(spoken_prompt)
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)
        pred_text = self.processor.batch_decode(
            pred_ids, 
            skip_special_tokens=True,
            decode_with_timestamps=False
        )[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text


@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="microsoft/Phi-3-mini-4k-instruct",
        metadata={
            "help": "The pretrained language model to use. Default is 'microsoft/Phi-3-mini-4k-instruct'."
        }
    )
    lm_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        }
    )
    user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        }
    )
    init_chat_role: str = field(
        default=None,
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        }
    )
    init_chat_prompt: str = field(
        default="You are a helpful AI assistant.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        }
    )
    lm_gen_max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 128."}
    )
    lm_gen_temperature: float = field(
        default=0.0,
        metadata={"help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."}
    )
    lm_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."}
    )
    chat_size: int = field(
        default=1,
        metadata={"help": "Number of interactions assitant-user to keep for the chat. None for no limitations."}
    )


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
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)
        self.pipe = pipeline( 
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
        ) 
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs
        }

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(f"An initial promt needs to be specified when setting init_chat_role.")
            self.chat.init_chat(
                {"role": init_chat_role, "content": init_chat_prompt}
            )
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Write me a poem about Machine Learning."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": self.gen_kwargs["max_new_tokens"],
            "max_new_tokens": self.gen_kwargs["max_new_tokens"],
            **self.gen_kwargs
        }

        n_steps = 2

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            thread = Thread(target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs)
            thread.start()
            for _ in self.streamer: 
                pass    
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, prompt):
        logger.debug("infering language model...")

        self.chat.append(
            {"role": self.user_role, "content": prompt}
        )
        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                yield(sentences[0])
                printable_text = new_text

        self.chat.append(
            {"role": "assistant", "content": generated_text}
        )

        # don't forget last sentence
        yield printable_text


@dataclass
class ParlerTTSHandlerArguments:
    tts_model_name: str = field(
        default="ylacombe/parler-tts-mini-jenny-30H",
        metadata={
            "help": "The pretrained TTS model to use. Default is 'ylacombe/parler-tts-mini-jenny-30H'."
        }
    )
    tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    tts_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        }
    )
    tts_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        }
    )
    tts_gen_min_new_tokens: int = field(
        default=None,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 10, which corresponds to ~0.1 secs"}
    )
    tts_gen_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 256, which corresponds to ~6 secs"}
    )
    description: str = field(
        default=(
            "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
            "She speaks very fast."
        ),
        metadata={
            "help": "Description of the speaker's voice and speaking style to guide the TTS model."
        }
    )
    play_steps_s: float = field(
        default=0.2,
        metadata={
            "help": "The time interval in seconds for playing back the generated speech in steps. Default is 0.5 seconds."
        }
    )
    max_prompt_pad_length: int = field(
        default=8,
        metadata={
            "help": "When using compilation, the prompt as to be padded to closest power of 2. This parameters sets the maximun power of 2 possible."
        }
    ) 


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
            play_steps_s=1
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
            model_name,
            torch_dtype=self.torch_dtype
        ).to(device)
        
        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)

        if self.compile_mode not in (None, "default"):
            logger.warning("Torch compilation modes that captures CUDA graphs are not yet compatible with the STT part. Reverting to 'default'")
            self.compile_mode = "default"

        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)

        self.warmup()

    def prepare_model_inputs(
        self,
        prompt,
        max_length_prompt=50,
        pad=False,
    ):
        pad_args_prompt = {"padding": "max_length", "max_length": max_length_prompt} if pad else {}

        tokenized_description = self.description_tokenizer(self.description, return_tensors="pt")
        input_ids = tokenized_description.input_ids.to(self.device)
        attention_mask = tokenized_description.attention_mask.to(self.device)

        tokenized_prompt = self.prompt_tokenizer(prompt, return_tensors="pt", **pad_args_prompt)
        prompt_input_ids = tokenized_prompt.input_ids.to(self.device)
        prompt_attention_mask = tokenized_prompt.attention_mask.to(self.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            **self.gen_kwargs
        }

        return gen_kwargs
    
    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture 
        n_steps = 1 if self.compile_mode == "default" else 2

        torch.cuda.synchronize()
        start_event.record()
        if self.compile_mode:
            pad_lengths = [2**i for i in range(2, self.max_prompt_pad_length)]
            for pad_length in pad_lengths[::-1]:
                model_kwargs = self.prepare_model_inputs(
                    "dummy prompt", 
                    max_length_prompt=pad_length,
                    pad=True
                )
                for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                logger.info(f"Warmed up length {pad_length} tokens!")
        else:
            model_kwargs = self.prepare_model_inputs("dummy prompt")
            for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                
        end_event.record() 
        torch.cuda.synchronize()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")


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

        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=self.play_steps)
        tts_gen_kwargs = {
            "streamer": streamer,
            **tts_gen_kwargs
        }
        torch.manual_seed(0)
        thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
        thread.start()

        for i, audio_chunk in enumerate(streamer):
            if i == 0:
                logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
            audio_chunk = np.int16(audio_chunk * 32767)
            yield audio_chunk

        self.should_listen.set()


def prepare_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """

    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1:]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def main():
    parser = HfArgumentParser((
        ModuleArguments,
        SocketReceiverArguments, 
        SocketSenderArguments,
        VADHandlerArguments,
        WhisperSTTHandlerArguments,
        LanguageModelHandlerArguments,
        ParlerTTSHandlerArguments,
    ))

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
            parler_tts_handler_kwargs,
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
            parler_tts_handler_kwargs,
        ) = parser.parse_args_into_dataclasses()

    # 1. Handle logger
    global logger
    logging.basicConfig(
        level=module_kwargs.log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if module_kwargs.log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

    # 2. Prepare each part's arguments
    prepare_args(whisper_stt_handler_kwargs, "stt")
    prepare_args(language_model_handler_kwargs, "lm")
    prepare_args(parler_tts_handler_kwargs, "tts") 

    # 3. Build the pipeline
    stop_event = Event()
    # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
    should_listen = Event() 
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue() 
    text_prompt_queue = Queue()
    lm_response_queue = Queue()
    
    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )
    stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        setup_kwargs=vars(whisper_stt_handler_kwargs),
    )
    lm = LanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs=vars(language_model_handler_kwargs),
    )
    tts = ParlerTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(parler_tts_handler_kwargs),
    )  

    recv_handler = SocketReceiver(
        stop_event, 
        recv_audio_chunks_queue, 
        should_listen,
        host=socket_receiver_kwargs.recv_host,
        port=socket_receiver_kwargs.recv_port,
        chunk_size=socket_receiver_kwargs.chunk_size,
    )

    send_handler = SocketSender(
        stop_event, 
        send_audio_chunks_queue,
        host=socket_sender_kwargs.send_host,
        port=socket_sender_kwargs.send_port,
        )

    # 4. Run the pipeline
    try:
        pipeline_manager = ThreadManager([vad, tts, lm, stt, recv_handler, send_handler])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()
    
if __name__ == "__main__":
    main()
