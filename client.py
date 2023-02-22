"""
The client module of the application responsible for recording and sending real time audio to the server.
"""

import socket
import pyaudio
import sounddevice as sd
import asyncio
import threading


def send_audio()-> None:
    """
    Records and sends the audio data to the server.

    Parameters
    ----------
    None

    Returns
    -------
    None
    
    """

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    HOST = '127.0.0.1' # Server IP address
    PORT = 8080 # Server port number

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    while True:
        data = stream.read(CHUNK)
        client_socket.sendall(data)

    client_socket.close()


if __name__ == "__main__":
    send_audio()
