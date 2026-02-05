from fastapi import FastAPI, Depends, HTTPException
from app.schemas import AudioRequest, DetectionResponse
from app.auth import verify_api_key
from app.audio_utils import decode_base64_audio_to_wav
from app.detector import detect_ai_voice

app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0"
)

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "AI Generated Voice Detection API is live"
    }

@app.post("/detect", response_model=DetectionResponse)
def detect_voice(
    request: AudioRequest,
    _: str = Depends(verify_api_key)
):
    if request.audio_format.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 audio format is supported"
        )

    try:
        wav_path = decode_base64_audio_to_wav(request.audio_base64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted audio input"
        )

    classification, confidence, explanation = detect_ai_voice(
        wav_path,
        request.language.lower()
    )

    return DetectionResponse(
        classification=classification,
        confidence=confidence,
        explanation=explanation
    )
