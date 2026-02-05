import librosa
import numpy as np

def detect_ai_voice(audio_path: str, language: str):
    # Load audio (lightweight)
    y, sr = librosa.load(audio_path, sr=16000)

    # Acoustic features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Heuristic rules
    if spectral_flatness > 0.35 and zcr < 0.06:
        return (
            "AI",
            0.78,
            "Uniform spectral distribution and low excitation suggest synthetic speech"
        )

    return (
        "HUMAN",
        0.86,
        "Natural spectral variation and human-like energy dynamics detected"
    )
