import socket
import sounddevice as sd
import numpy as np
import threading
from queue import Queue

CHUNK = 1024
CHANNELS = 1
SEND_RATE = 16000
RECV_RATE = 44100

HOST = '172.16.128.13'  
PORT = 12345   
RECV_PORT = 12346    

send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_socket.connect((HOST, PORT))

recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
recv_socket.connect((HOST, RECV_PORT))

print("Recording and streaming...")

stop_event = threading.Event()
recv_queue = Queue()
send_queue = Queue()

def callback_recv(outdata, frames, time, status): 
    if not recv_queue.empty():
        data = recv_queue.get()
        outdata[:len(data)] = data 
        outdata[len(data):] = b'\x00' * (len(outdata) - len(data)) 
    else:
        outdata[:] = b'\x00' * len(outdata) 

def callback_send(indata, frames, time, status):
    if recv_queue.empty():
        data = bytes(indata)
        send_queue.put(data)

def send(stop_event, send_queue):
    while not stop_event.is_set():
        data = send_queue.get()
        send_socket.sendall(data)
       
def recv(stop_event, recv_queue):

    def receive_full_chunk(conn, chunk_size):
        data = b''
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                return None  # Connection has been closed
            data += packet
        return data

    while not stop_event.is_set():
        data = receive_full_chunk(recv_socket, CHUNK * 2) 
        if data:
            recv_queue.put(data)

try: 
    send_stream = sd.RawInputStream(samplerate=SEND_RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK, callback=callback_send)
    recv_stream = sd.RawOutputStream(samplerate=RECV_RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK, callback=callback_recv)
    threading.Thread(target=send_stream.start).start()
    threading.Thread(target=recv_stream.start).start()

    send_thread = threading.Thread(target=send, args=(stop_event, send_queue))
    send_thread.start()
    recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue))
    recv_thread.start()
    
    input("Press Enter to stop...")

except KeyboardInterrupt:
    print("Finished streaming.")

finally:
    stop_event.set()
    recv_thread.join()
    print("1")

    send_thread.join()
    print("2")

    send_socket.close()
    recv_socket.close()
    print("Connection closed.")