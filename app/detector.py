import librosa
import numpy as np

MAX_DURATION = 4.0  # seconds

def detect_ai_voice(audio_path: str, language: str):
    y, sr = librosa.load(audio_path, sr=16000, duration=MAX_DURATION)

    if y is None or len(y) == 0:
        return (
            "AI",
            0.5,
            "Audio signal could not be reliably decoded"
        )

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    # NEW: pitch variability
    pitches, voiced_flags, _ = librosa.pyin(
        y,
        fmin=80,
        fmax=400,
        sr=sr
    )

    pitch_var = 0.0
    if pitches is not None:
        valid_pitches = pitches[~np.isnan(pitches)]
        if len(valid_pitches) > 10:
            pitch_var = np.std(valid_pitches)

    # AI-like conditions
    if (
        spectral_flatness > 0.30 and
        zcr < 0.08 and
        pitch_var < 15
    ):
        return (
            "AI",
            0.82,
            "Low pitch variability and uniform spectral patterns suggest synthetic speech"
        )

    return (
        "HUMAN",
        0.86,
        "Natural pitch variation and irregular spectral energy indicate human speech"
    )
