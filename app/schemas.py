from pydantic import BaseModel, Field

class AudioRequest(BaseModel):
    language: str = Field(..., example="english")
    audio_format: str = Field(..., example="mp3")
    audio_base64: str = Field(..., description="Base64 encoded MP3 audio")

class DetectionResponse(BaseModel):
    classification: str
    confidence: float
    explanation: str
