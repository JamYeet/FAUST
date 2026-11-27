import os
def add_nvidia_dll_paths():
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return  # packages not installed, e.g. on Linux/CPU — skip
    for pkg in (nvidia.cublas, nvidia.cudnn):
        bin_dir = os.path.join(list(pkg.__path__)[0], "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)                              # for Python's own loading
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]  # for libraries loading DLLs themselves

add_nvidia_dll_paths() 

import queue
import numpy as np
import sounddevice as sd
import torch

from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000   # Silero VAD expects 16 kHz audio
CHUNK = 512           # Silero needs exactly 512 samples per chunk (~32 ms)

audio_q = queue.Queue()


def callback(indata, frames, time, status):
    audio_q.put(indata[:, 0].copy()) 

def load_whisper():
    model_size = "small"
    
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    return model

def transcribe(model, audio):
    
    segments, info = model.transcribe(audio, language="en")
    joined = ""
    for s in segments:
        joined += s.text 
    return joined.strip()             
    

def main():
    vad_model = load_silero_vad()
    vad = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
    whisper_model = load_whisper()

    speaking = False                  
    chunks = []                     
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK, callback=callback):
        print("Listening... (Ctrl+C to stop)")
        while True:
            chunk = audio_q.get()
            result = vad(torch.from_numpy(chunk))

            if result and 'start' in result:
                speaking = True
                chunks = []          
                print("Speech started")

            if speaking:
                chunks.append(chunk) 

            if result and 'end' in result:
                speaking = False
                audio = np.concatenate(chunks)
                text = transcribe(whisper_model, audio)
                print(f"You said: {text}")

if __name__ == "__main__":
    main()