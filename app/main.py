from fastapi import FastAPI, Header, Request
import base64, os
from app.detector import detect_ai_voice

app = FastAPI()

@app.post("/detect")
async def detect_voice(
    request: Request,
    x_api_key: str = Header(...)
):
    if x_api_key != "hackathon_secret_key":
        return {"detail": "Invalid API key"}

    data = {}

    # 1️⃣ Try JSON first (hackathon tester)
    try:
        data = await request.json()
    except:
        pass

    # 2️⃣ If JSON empty, try form-data
    if not data:
        form = await request.form()
        data = dict(form)

    # 3️⃣ Read fields (support both cases)
    language = data.get("language")
    audio_format = data.get("audio_format") or data.get("audioFormat")
    audio_base64 = data.get("audio_base64") or data.get("audioBase64")

    if not language or not audio_format or not audio_base64:
        return {
            "detail": "Missing one or more required fields: language, audio_format, audio_base64",
            "received_keys": list(data.keys())
        }

    audio_bytes = base64.b64decode(audio_base64)
    temp_file = f"temp_audio.{audio_format}"

    with open(temp_file, "wb") as f:
        f.write(audio_bytes)

    classification, confidence, explanation = detect_ai_voice(
        temp_file, language
    )

    os.remove(temp_file)

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }
