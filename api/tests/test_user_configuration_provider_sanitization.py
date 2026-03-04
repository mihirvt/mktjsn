from api.schemas.user_configuration import UserConfiguration
from api.services.configuration.registry import ServiceProviders


def test_invalid_provider_is_removed_per_service_not_whole_config():
    cfg = UserConfiguration.model_validate(
        {
            "llm": {
                "provider": ServiceProviders.OPENAI.value,
                "model": "gpt-4.1",
                "api_key": "llm-key",
            },
            "stt": {
                "provider": ServiceProviders.SMALLEST_AI.value,
                "model": "legacy-invalid-stt-model",
                "api_key": "stt-key",
            },
            "tts": {
                "provider": ServiceProviders.SMALLEST_AI.value,
                "model": "lightning-v3.1",
                "api_key": "tts-key",
                "voice": "aisha",
                "language": "hi",
            },
        }
    )

    assert cfg.llm is not None
    assert cfg.llm.provider == ServiceProviders.OPENAI
    assert cfg.tts is not None
    assert cfg.tts.provider == ServiceProviders.SMALLEST_AI
    # Invalid STT provider is dropped instead of invalidating the full config.
    assert cfg.stt is None


def test_legacy_gemini_provider_is_removed():
    cfg = UserConfiguration.model_validate(
        {
            "tts": {
                "provider": "gemini",
                "model": "ignored",
                "api_key": "ignored",
            }
        }
    )

    assert cfg.tts is None
