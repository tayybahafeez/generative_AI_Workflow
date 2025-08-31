# speech_to_text.py
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile
import os

model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    print("🎙️ Recording... Speak now")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return recording, fs

def transcribe_from_mic():
    audio, fs = record_audio()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        scipy.io.wavfile.write(f.name, fs, audio)
        result = model.transcribe(f.name)
        os.remove(f.name)
        return result["text"]

def transcribe_from_file(file_path):
    print(f"📁 Transcribing file: {file_path}")
    result = model.transcribe(file_path)
    return result["text"]
