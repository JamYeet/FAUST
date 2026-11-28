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
import math

from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
from collections import deque


# Constants
SAMPLE_RATE = 16000   # Silero VAD expects 16 kHz audio
CHUNK = 512           # Silero needs exactly 512 samples per chunk (~32 ms)
PRE_ROLL_MS = 300
PRE_ROLL_CHUNKS = math.ceil(PRE_ROLL_MS / (CHUNK / SAMPLE_RATE * 1000)) # 10
MIN_SPEECH_S = 0.2
NO_SPEECH_THRESHOLD = 0.6
MIN_TOTAL_S = PRE_ROLL_CHUNKS * CHUNK / SAMPLE_RATE + MIN_SPEECH_S 
INITIAL_PROMPT = "Hey Faust, what's the time?"


audio_q = queue.Queue()


def callback(indata, frames, time, status):
    audio_q.put(indata[:, 0].copy()) 

def load_whisper():
    model_size = "small"
    
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    return model

def transcribe(model, audio):
    

    segments, info = model.transcribe(audio, language="en", initial_prompt=INITIAL_PROMPT)
    joined = ""
    for s in segments:
        if s.no_speech_prob < NO_SPEECH_THRESHOLD:
            joined += s.text 
    return joined.strip()             
    

def main():
    vad_model = load_silero_vad()
    vad = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
    whisper_model = load_whisper()

    speaking = False                  
    chunks = []  
    pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)
                       
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK, callback=callback):
        print("Listening... (Ctrl+C to stop)")
        while True:
            chunk = audio_q.get()
            result = vad(torch.from_numpy(chunk))

            if result and 'start' in result: # On start
                speaking = True
                chunks = list(pre_roll)       
                print("Speech started") 

            if speaking: # If speaking
                chunks.append(chunk) 
            else:
                pre_roll.append(chunk)

            if result and 'end' in result: # On end
                speaking = False
                pre_roll.clear()
                

                audio = np.concatenate(chunks)
                audio_length = len(audio) / SAMPLE_RATE
                
                if audio_length < MIN_TOTAL_S:
                    print("Ignored", round(audio_length, 2), "s length clip")
                    continue
                
                text = transcribe(whisper_model, audio)
                if not text:
                    print("No speech detected")
                else:
                    print(f"You said: {text}")


if __name__ == "__main__":
    main()