from datetime import datetime
from typing import Any
from pydantic import BaseModel, model_validator

class DummyConfig(BaseModel):
    provider: str

class UserConfiguration(BaseModel):
    tts: DummyConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def strip(cls, data: Any) -> Any:
        print("Running validator, data:", data)
        if isinstance(data, dict):
            if "tts" in data and isinstance(data["tts"], dict):
                if data["tts"].get("provider") == "gemini":
                    data["tts"] = None
        return data

try:
    obj = UserConfiguration.model_validate({"tts": {"provider": "gemini"}})
    print("Success:", obj)
except Exception as e:
    print("Error:", e)
