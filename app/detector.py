import numpy as np
import librosa
import whisper
import os

# Lazy-loaded model (IMPORTANT for low-memory environments)
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


def detect_ai_voice(wav_path: str, language: str):
    waveform, sr = librosa.load(wav_path, sr=16000)

    spectral_flatness = float(
        np.mean(librosa.feature.spectral_flatness(y=waveform))
    )

    zero_crossing = float(
        np.mean(librosa.feature.zero_crossing_rate(waveform))
    )

    whisper_model = get_whisper_model()

    result = whisper_model.transcribe(
        wav_path,
        language=language,
        fp16=False
    )

    if not result.get("segments"):
        os.remove(wav_path)
        return "HUMAN", 0.5, "Insufficient speech content detected"

    avg_logprob = np.mean(
        [seg["avg_logprob"] for seg in result["segments"]]
    )

    ai_score = 0.0

    if spectral_flatness > 0.22:
        ai_score += 0.35

    if zero_crossing < 0.05:
        ai_score += 0.25

    if avg_logprob > -0.25:
        ai_score += 0.40

    ai_score = min(ai_score, 1.0)

    os.remove(wav_path)

    if ai_score >= 0.55:
        return (
            "AI_GENERATED",
            round(ai_score, 2),
            "Highly uniform spectral patterns and synthetic speech consistency detected"
        )
    else:
        return (
            "HUMAN",
            round(1 - ai_score, 2),
            "Natural speech variability and human-like acoustic features detected"
        )
