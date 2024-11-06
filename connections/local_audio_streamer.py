import threading
import sounddevice as sd
import numpy as np
import time
import logging
from pynput import keyboard

logger = logging.getLogger(__name__)

class LocalAudioStreamer:
    def __init__(
        self,
        input_queue,
        output_queue,
        list_play_chunk_size=512,
    ):
        self.list_play_chunk_size = list_play_chunk_size
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.input_device = None
        self.output_device = None
        self.stream = None
        self.keyboard_listener = None

    def list_devices(self):
        devices = sd.query_devices()
        print("Périphériques audio disponible : ")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} ({'input' if device['max_input_channels'] > 0 else 'output'})")

    def select_device(self, device_type):
        self.list_devices()
        while True:
            try:
                choice = int(input(f"Select {device_type} device number: "))
                device = sd.query_devices(choice)
                if (device_type == 'input' and device['max_input_channels'] > 0) or \
                   (device_type == 'output' and device['max_output_channels'] > 0):
                    return choice
                else:
                    print(f"Invalid {device_type} device. Please try again.")
            except (ValueError, sd.PortAudioError):
                print("Invalid input. Please enter a number from the list.")

    def run(self):
        self.input_device = self.select_device('input')
        self.output_device = self.select_device('output')

        def callback(indata, outdata, frames, time, status):
            if not self.pause_event.is_set():
                if self.output_queue.empty():
                    self.input_queue.put(indata.copy())
                    outdata[:] = 0 * outdata
                else:
                    outdata[:] = self.output_queue.get()[:, np.newaxis]
            else:
                outdata[:] = 0 * outdata  # Output silence when paused

        logger.debug("Selected devices:")
        logger.debug(f"Input: {sd.query_devices(self.input_device)['name']}")
        logger.debug(f"Output: {sd.query_devices(self.output_device)['name']}")

        self.stream = sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
            device=(self.input_device, self.output_device)
        )

        def on_press(key):
            try:
                if key == keyboard.Key.cmd and keyboard.KeyCode.from_char('s'):
                    if self.pause_event.is_set():
                        self.resume()
                    else:
                        self.pause()
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

        print("Poly est en cours d'exécution… Appuyez sur Cmd pour mettre le modèle en pause, ou Ctrl+C pour l'arrêter.")

        with self.stream:
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")

        self.keyboard_listener.stop()

    def pause(self):
        self.pause_event.set()
        logger.info("Poly est en pause")
        print("Poly est en pause")

    def resume(self):
        self.pause_event.clear()
        logger.info("Poly t'écoute…")
        print("Poly t'écoute…")

    def stop(self):
        self.stop_event.set()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.stream:
            self.stream.stop()