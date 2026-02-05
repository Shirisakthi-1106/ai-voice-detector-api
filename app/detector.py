import numpy as np
import librosa
import whisper
import os
import re
from collections import Counter

# Load Whisper model once
whisper_model = whisper.load_model("base")

def detect_repetition(text: str) -> float:
    """
    Detects repetitive synthetic speech patterns.
    Returns repetition score between 0 and 1.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < 10:
        return 0.0

    counts = Counter(words)
    most_common_ratio = counts.most_common(1)[0][1] / len(words)

    return most_common_ratio  # higher = more repetitive

def detect_ai_voice(wav_path: str, language: str):

    # Load audio
    waveform, sr = librosa.load(wav_path, sr=16000)

    # ---- Acoustic features ----
    spectral_flatness = float(
        np.mean(librosa.feature.spectral_flatness(y=waveform))
    )

    zero_crossing = float(
        np.mean(librosa.feature.zero_crossing_rate(waveform))
    )

    # ---- Whisper transcription ----
    result = whisper_model.transcribe(
        wav_path,
        language=language,
        fp16=False
    )

    if not result.get("segments"):
        os.remove(wav_path)
        return "HUMAN", 0.5, "Insufficient speech content detected"

    full_text = " ".join(seg["text"] for seg in result["segments"])
    avg_logprob = np.mean(
        [seg["avg_logprob"] for seg in result["segments"]]
    )

    repetition_score = detect_repetition(full_text)

    # ---- AI SCORING (AGGRESSIVE DEMO MODE) ----
    ai_score = 0.0

    # Acoustic uniformity
    if spectral_flatness > 0.20:
        ai_score += 0.25

    # Low articulation noise
    if zero_crossing < 0.06:
        ai_score += 0.20

    # Whisper is TOO confident → synthetic
    if avg_logprob > -0.30:
        ai_score += 0.30

    # Repetition = classic TTS giveaway
    if repetition_score > 0.12:
        ai_score += 0.40

    ai_score = min(ai_score, 1.0)

    os.remove(wav_path)

    if ai_score >= 0.50:
        return (
            "AI_GENERATED",
            round(ai_score, 2),
            "Repetitive linguistic patterns and synthetic speech consistency detected"
        )
    else:
        return (
            "HUMAN",
            round(1 - ai_score, 2),
            "Natural speech variability and human-like acoustic features detected"
        )
