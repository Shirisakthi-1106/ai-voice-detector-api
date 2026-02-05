from fastapi import FastAPI, Header, Form, Request
import base64
import os
from app.detector import detect_ai_voice

app = FastAPI()

@app.post("/detect")
async def detect_voice(
    request: Request,
    x_api_key: str = Header(...)
):
    if x_api_key != "hackathon_secret_key":
        return {"detail": "Invalid API key"}

    form = await request.form()

    # Accept both snake_case and camelCase
    language = form.get("language")
    audio_format = form.get("audio_format") or form.get("audioFormat")
    audio_base64 = form.get("audio_base64") or form.get("audioBase64")

    if not language or not audio_format or not audio_base64:
        return {
            "detail": "Missing one or more required fields: language, audio_format, audio_base64"
        }

    # Decode Base64
    audio_bytes = base64.b64decode(audio_base64)

    temp_path = f"temp_audio.{audio_format}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    classification, confidence, explanation = detect_ai_voice(
        temp_path, language
    )

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }
