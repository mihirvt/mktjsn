from datetime import datetime
from typing import Any

from pydantic import BaseModel, model_validator

from api.services.configuration.registry import (
    EmbeddingsConfig,
    LLMConfig,
    STTConfig,
    TTSConfig,
)


class UserConfiguration(BaseModel):
    llm: LLMConfig | None = None
    stt: STTConfig | None = None
    tts: TTSConfig | None = None
    embeddings: EmbeddingsConfig | None = None
    test_phone_number: str | None = None
    timezone: str | None = None
    last_validated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def strip_deprecated_providers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for service in ["llm", "stt", "tts", "embeddings"]:
                if service in data and isinstance(data[service], dict):
                    if data[service].get("provider") == "gemini":
                        # If a legacy 'gemini' provider is found, remove it so it doesn't break Pydantic validation
                        # The system will automatically fall back to the defaults instead of crashing
                        data[service] = None
            # Convert boolean enhancement to int for smallest_ai TTS configs
            if "tts" in data and isinstance(data.get("tts"), dict):
                enhancement = data["tts"].get("enhancement")
                if isinstance(enhancement, bool):
                    data["tts"]["enhancement"] = int(enhancement)
        return data
