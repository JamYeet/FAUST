import queue
import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000   # Silero VAD expects 16 kHz audio
CHUNK = 512           # Silero needs exactly 512 samples per chunk (~32 ms)

audio_q = queue.Queue()

def callback(indata, frames, time, status):
    # sounddevice calls this automatically every time a new chunk arrives.
    # Keep it tiny: just copy the audio into the queue and return.
    audio_q.put(indata[:, 0].copy())   # [:, 0] = first (mono) channel

def main():
    model = load_silero_vad()
    vad = VADIterator(model, sampling_rate=SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK, callback=callback):
        print("Listening... (Ctrl+C to stop)")
        while True:
            chunk = audio_q.get()                     # wait for the next chunk
            result = vad(torch.from_numpy(chunk))     # None, {'start': ...} or {'end': ...}
            if result:
                if 'start' in result:
                    print("Speech started")
                elif 'end' in result: 
                    print("Speech stopped")

if __name__ == "__main__":
    main()