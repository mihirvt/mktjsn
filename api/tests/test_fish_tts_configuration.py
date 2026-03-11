from api.schemas.user_configuration import UserConfiguration
from api.services.configuration.registry import ServiceProviders


def test_fish_tts_configuration_parses_and_preserves_fields():
    cfg = UserConfiguration.model_validate(
        {
            "tts": {
                "provider": ServiceProviders.FISH.value,
                "model": "s1",
                "voice": "bf322df2096a46f18c579d0baa36f41d",
                "api_key": "fish-key",
                "latency": "balanced",
                "top_p": "0.6",
                "temperature": "0.4",
                "chunk_length": "180",
                "speed": "1.1",
                "volume": "2",
                "telephony_sample_rate": "8000",
                "web_sample_rate": "16000",
            }
        }
    )

    assert cfg.tts is not None
    assert cfg.tts.provider == ServiceProviders.FISH
    assert cfg.tts.model == "s1"
    assert cfg.tts.voice == "bf322df2096a46f18c579d0baa36f41d"
    assert cfg.tts.top_p == 0.6
    assert cfg.tts.temperature == 0.4
    assert cfg.tts.chunk_length == 180
    assert cfg.tts.speed == 1.1
    assert cfg.tts.volume == 2
    assert cfg.tts.telephony_sample_rate == 8000
    assert cfg.tts.web_sample_rate == 16000
