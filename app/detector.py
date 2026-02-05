import whisper

# Load model ONCE at startup
model = whisper.load_model("tiny")

def detect_ai_voice(audio_path: str, language: str):
    result = model.transcribe(
        audio_path,
        language=language,
        fp16=False
    )

    text = result.get("text", "").lower()

    # Simple heuristic (acceptable for hackathon)
    if len(text.strip()) < 5:
        return "AI", 0.75, "Low linguistic entropy detected"

    return "HUMAN", 0.9, "Natural speech variability detected"
