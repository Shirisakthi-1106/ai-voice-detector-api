import base64
import io
import numpy as np
import soundfile as sf
import tempfile
import librosa
import os

def decode_base64_audio_to_wav(audio_base64: str) -> str:
    """
    Decodes Base64 MP3 audio, converts to 16kHz mono WAV,
    saves to a temporary file, and returns the file path.
    """

    # Decode Base64
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    # Read audio
    waveform, sr = sf.read(audio_buffer)

    # Convert to mono if needed
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample to 16kHz (Whisper requirement)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Save to temporary WAV file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, waveform.astype("float32"), sr)

    return tmp_file.name
