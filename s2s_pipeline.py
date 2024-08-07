import logging
import socket
import threading
from threading import Thread, Event
from queue import Queue
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from nltk.tokenize import sent_tokenize
from rich.console import Console
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    AutoTokenizer, 
    pipeline, 
    TextIteratorStreamer
)
from parler_tts import ParlerTTSForConditionalGeneration

# Local module imports
from utils import VADIterator, int2float, ParlerTTSStreamer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
console = Console()


class ThreadManager:
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

pipeline_start = None

class BaseHandler:
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


class SocketReceiver:
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
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(1)
        self.conn, _ = self.socket.accept()
        logger.debug("receiver connected")

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
        logger.debug("Receiver closed")
                

class SocketSender:
    def __init__(
        self, 
        stop_event,
        queue_in,
        host='0.0.0.0', 
        port=12346
    ):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(1)
        self.conn, _ = self.socket.accept()
        logger.debug("sender connected")

    def run(self):
        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            self.conn.sendall(audio_chunk)
            if isinstance(audio_chunk, bytes) and audio_chunk == b'END':
                break
        self.conn.close()
        logger.debug("Sender closed")


class VADHandler(BaseHandler):
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
        self._should_listen = should_listen
        self._sample_rate = sample_rate
        self._min_silence_ms = min_silence_ms
        self._min_speech_ms = min_speech_ms
        self._max_speech_ms = max_speech_ms
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
        if vad_output is not None:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self._sample_rate * 1000
            if duration_ms < self._min_speech_ms or duration_ms > self._max_speech_ms:
                logger.debug(f"audio input of duration: {len(array) / self._sample_rate}s, skipping")
            else:
                self._should_listen.clear()
                logger.debug("Stop listening")
                yield array


class WhisperSTTProcessor(BaseHandler):
    def setup(
            self,
            model_name="distil-whisper/distil-large-v3",
            device="cuda",  
            torch_dtype=torch.float16,  
            gen_kwargs={}
        ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.gen_kwargs = gen_kwargs

    def process(self, spoken_prompt):
        global pipeline_start
        pipeline_start = perf_counter()
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)
        logger.debug("infering whisper...")
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)
        pred_text = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True,
            decode_with_timestamps=False
        )[0]
        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        yield pred_text


class LanguageModelHandler(BaseHandler):
    def setup(
            self,
            model_name="microsoft/Phi-3-mini-4k-instruct",
            device="cuda", 
            torch_dtype=torch.float16,
            gen_kwargs={},
            user_role="user",
            init_chat_role="system", 
            init_chat_prompt="You are a helpful AI assistant.",
        ):
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
        self.chat = [
            {"role": init_chat_role, "content": init_chat_prompt}
        ]
        self.gen_kwargs = {
            "streamer": self.streamer,
            **gen_kwargs
        }
        self.user_role = user_role

    def process(self, prompt):
        self.chat.append(
            {"role": self.user_role, "content": prompt}
        )
        thread = Thread(target=self.pipe, args=(self.chat,), kwargs=self.gen_kwargs)
        thread.start()
        generated_text, printable_text = "", ""
        logger.debug("infering language model...")
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


class ParlerTTSProcessor(BaseHandler):
    def setup(
            self,
            should_listen,
            model_name="ylacombe/parler-tts-mini-jenny-30H",
            device="cuda", 
            torch_dtype=torch.float32,
            gen_kwargs={},
            description=(
                "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
                "She speaks very fast."
            ),
            play_steps_s=0.5
        ):
        self._should_listen = should_listen
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)
        self.device = device
        self.torch_dtype = torch_dtype

        tokenized_description = self.description_tokenizer(description, return_tensors="pt")
        input_ids = tokenized_description.input_ids.to(self.device)
        attention_mask = tokenized_description.attention_mask.to(self.device)

        self.gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **gen_kwargs
        }
        
        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)

    def process(self, llm_sentence):
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        tokenized_prompt = self.prompt_tokenizer(llm_sentence, return_tensors="pt")
        prompt_input_ids = tokenized_prompt.input_ids.to(self.device)
        prompt_attention_mask = tokenized_prompt.attention_mask.to(self.device)

        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=self.play_steps)
        tts_gen_kwargs = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "streamer": streamer,
            **self.gen_kwargs
        }

        torch.manual_seed(0)
        thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
        thread.start()

        for i, audio_chunk in enumerate(streamer):
            if i == 0:
                logger.debug(f"time to first audio: {perf_counter() - pipeline_start:.3f}")
            audio_chunk = np.int16(audio_chunk * 32767)
            yield audio_chunk

        self._should_listen.set()


def main():

    device = "cuda:1"

    stt_gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "return_timestamps": False,
        "task": "transcribe",
        "language": "en",
    }

    llm_gen_kwargs = { 
        "max_new_tokens": 128, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    tts_gen_kwargs = {
        "min_new_tokens": 10,
        "temperature": 1.0,
        "do_sample": True,
    }

    stop_event = Event()
    should_listen = Event()

    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue() 
    text_prompt_queue = Queue()
    llm_response_queue = Queue()
    
    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,)
    )
    stt = WhisperSTTProcessor(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        setup_kwargs={
            "device": device,
            "gen_kwargs": stt_gen_kwargs,
        },
    )
    llm = LanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=llm_response_queue,
        setup_kwargs={
            "device": device,
            "gen_kwargs": llm_gen_kwargs,
        },
    )
    tts = ParlerTTSProcessor(
        stop_event,
        queue_in=llm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs={
            "device": device,
            "gen_kwargs": tts_gen_kwargs
        },
    )  

    recv_handler = SocketReceiver(
        stop_event, 
        recv_audio_chunks_queue, 
        should_listen,
    )

    send_handler = SocketSender(
        stop_event, 
        send_audio_chunks_queue,
    )

    try:
        pipeline_manager = ThreadManager([vad, tts, llm, stt, recv_handler, send_handler])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()

    
if __name__ == "__main__":
    main()
