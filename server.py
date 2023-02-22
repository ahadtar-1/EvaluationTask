"""
The server module of the application that receives the real time audio data and converts audio into text.
"""

import socket
import threading
import queue
import pyaudio
import numpy as np
import torch
from absl import app, flags
from rnnt.args import FLAGS
from rnnt.stream import PytorchStreamDecoder
from parts.features import AudioPreprocessing

#Arguments of PyTorch Stream Decoder
flags.DEFINE_string("flagfile", default="./flagfiles/E6D2_LARGE_Batch.txt", help="flag file")
flags.DEFINE_string("model_name", default="english_43_medium.pt", help="steps of checkpoint")
flags.DEFINE_integer('step_n_frame', 2, help='input frame(stacked)')
flags.DEFINE_enum('stream_decoder', 'torch', ['torch', 'openvino'], help='stream decoder implementation')
flags.DEFINE_string('url', 'https://www.youtube.com/watch?v=2EppLNonncc', help='youtube live link')
flags.DEFINE_integer('reset_step', 500, help='reset hidden state')
flags.DEFINE_string('path', None, help='path to .wav')


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.        
    """
        
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def receive_audio(client_socket : socket, q : queue)-> None:
    """
    Receives the audio data into text.

    Parameters
    ----------
    client_socket
        The socket through which the server receives the real time audio from the client.

    Returns
    -------
    None
    
    """
    
    transform = AudioPreprocessing(normalize='none', sample_rate=16000, window_size=0.02, 
    window_stride=0.015, features=80, n_fft=512, 
    feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False)
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    mfcc = []
    audio_data = b''
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        audio_data += data   
        if len(audio_data) == CHUNK * 2:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).reshape(-1, CHANNELS)
            print(audio_np, type(audio_np))
            audioo_np = audio_np[-2:]
            print(audioo_np, type(audioo_np))
            indata = np.concatenate(audio_np[-2:], axis=0)
            print(indata, type(indata))
            indataa = indata / (1<<16)
            print(indataa, type(indataa))
            waveform = transform(torch.from_numpy(indata.flatten()).float()).T
            mfcc.append(waveform)
            mfcc = torch.cat(mfcc).T
            print("MFCC:", mfcc, mfcc.shape, type(mfcc))
            #print(waveform, type(waveform))
            #final_waveform = waveform.unsqueeze(0)
            #print(final_waveform, type(final_waveform.shape), final_waveform.shape)
            seq = stream_decoder.decode(mfcc)
            print(seq)
            #audio_data = b''

    client_socket.close()


def main(argv : PytorchStreamDecoder)-> None:
    """
    Binds the server module to the an ip on the local machine and calls the 'receive audio' function.

    Parameters
    ----------
    PytorchStreamDecoder

    Returns
    -------
    None
    
    """
        
    global stream_decoder
    stream_decoder = PytorchStreamDecoder(FLAGS)
    
    HOST = '127.0.0.1' # Server IP address
    PORT = 8080 # Server port numbers    
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))    
    server_socket.listen()

    q = queue.Queue()    
    
    while True:
        print("Working")
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        client_thread = threading.Thread(target = receive_audio, args = (client_socket, q), daemon=True)
        client_thread.start()

if __name__ == "__main__":
    app.run(main)
