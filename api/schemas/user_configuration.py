from datetime import datetime
from typing import Any

from pydantic import BaseModel, model_validator

from api.services.configuration.registry import (
    EmbeddingsConfig,
    LLMConfig,
    REGISTRY,
    STTConfig,
    ServiceType,
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
            service_to_type = {
                "llm": ServiceType.LLM,
                "stt": ServiceType.STT,
                "tts": ServiceType.TTS,
                "embeddings": ServiceType.EMBEDDINGS,
            }
            for service in ["llm", "stt", "tts", "embeddings"]:
                if service in data and isinstance(data[service], dict):
                    provider = data[service].get("provider")
                    if provider == "gemini":
                        # If a legacy 'gemini' provider is found, remove it so it doesn't break Pydantic validation
                        # The system will automatically fall back to the defaults instead of crashing
                        data[service] = None
                        continue

                    if isinstance(provider, str):
                        service_type = service_to_type[service]
                        if provider not in REGISTRY[service_type]:
                            # Keep other service configs intact when one stale provider is present.
                            data[service] = None
                            continue

                    # Normalize API keys copied from dashboards/editors that may include
                    # invisible Unicode separators (e.g. U+2028) or trailing whitespace.
                    api_key = data[service].get("api_key")
                    if isinstance(api_key, str):
                        normalized_key = (
                            api_key.replace("\u2028", "")
                            .replace("\u2029", "")
                            .replace("\r", "")
                            .replace("\n", "")
                            .strip()
                        )
                        data[service]["api_key"] = normalized_key
            # Convert boolean enhancement to int for smallest_ai TTS configs
            if "tts" in data and isinstance(data.get("tts"), dict):
                enhancement = data["tts"].get("enhancement")
                if isinstance(enhancement, bool):
                    data["tts"]["enhancement"] = int(enhancement)

            # Coerce numeric fields that the UI may send as strings or NaN.
            # HTML form inputs produce strings; empty inputs produce "" which
            # fail Pydantic's ge/le float constraints.
            for service in ["llm", "tts", "stt"]:
                cfg = data.get(service)
                if not isinstance(cfg, dict):
                    continue
                for key, val in list(cfg.items()):
                    if isinstance(val, str) and key not in ("provider", "api_key", "model",
                            "voice", "language", "base_url", "reasoning_effort",
                            "operating_point", "thinking_level"):
                        try:
                            cfg[key] = float(val) if val else None
                        except (ValueError, TypeError):
                            pass
                    # Remove None / NaN values so Pydantic uses field defaults
                    if cfg.get(key) is None:
                        cfg.pop(key, None)
                    elif isinstance(cfg.get(key), float) and cfg[key] != cfg[key]:
                        # NaN check (NaN != NaN)
                        cfg.pop(key, None)
        return data
