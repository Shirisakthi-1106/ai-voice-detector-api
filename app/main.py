from fastapi import FastAPI, Header, Form
import base64

app = FastAPI()

@app.post("/detect")
def detect_voice(
    language: str = Form(...),
    audio_format: str = Form(...),
    audio_base64: str = Form(...),
    x_api_key: str = Header(...)
):
    if x_api_key != "hackathon_secret_key":
        return {"detail": "Invalid API key"}

    # decode base64 audio
    audio_bytes = base64.b64decode(audio_base64)

    # save temp file
    temp_path = "temp_audio." + audio_format
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # run your existing detection logic
    classification, confidence, explanation = detect_ai_voice(
        temp_path, language
    )

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation
    }
