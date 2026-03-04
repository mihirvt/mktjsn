import sys
from types import ModuleType, SimpleNamespace

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import (
    _parse_soniox_language_hints,
    create_stt_service,
)
from pipecat.transcriptions.language import Language


class DummySonioxInputParams:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model")
        self.language_hints = kwargs.get("language_hints")
        self.language_hints_strict = kwargs.get("language_hints_strict")


class DummySonioxSTTService:
    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.params = kwargs.get("params")
        self.sample_rate = kwargs.get("sample_rate")


def _mock_soniox_module(monkeypatch):
    module = ModuleType("pipecat.services.soniox.stt")
    module.SonioxInputParams = DummySonioxInputParams
    module.SonioxSTTService = DummySonioxSTTService
    monkeypatch.setitem(sys.modules, "pipecat.services.soniox.stt", module)


def _build_user_config(**stt_overrides):
    stt_defaults = {
        "provider": ServiceProviders.SONIOX.value,
        "model": "stt-rt-v4",
        "api_key": "test-soniox-key",
    }
    stt_defaults.update(stt_overrides)
    return SimpleNamespace(stt=SimpleNamespace(**stt_defaults))


def test_parse_soniox_language_hints_filters_invalid_values():
    hints = _parse_soniox_language_hints(["EN", "invalid", "es"])
    assert hints == [Language.EN, Language.ES]


def test_create_stt_service_passes_language_hints_and_strict(monkeypatch):
    _mock_soniox_module(monkeypatch)
    user_config = _build_user_config(
        language_hints=["en", "es"],
        language_hints_strict=True,
    )
    audio_config = SimpleNamespace(transport_in_sample_rate=16000)

    service = create_stt_service(user_config, audio_config)

    assert service.params.model == "stt-rt-v4"
    assert service.params.language_hints == [Language.EN, Language.ES]
    assert service.params.language_hints_strict is True


def test_create_stt_service_falls_back_to_legacy_language(monkeypatch):
    _mock_soniox_module(monkeypatch)
    user_config = _build_user_config(
        language_hints=[],
        language_hints_strict=True,
        language="hi",
    )
    audio_config = SimpleNamespace(transport_in_sample_rate=8000)

    service = create_stt_service(user_config, audio_config)

    assert service.params.language_hints == [Language.HI]
    assert service.params.language_hints_strict is True


def test_create_stt_service_omits_strict_without_valid_hints(monkeypatch):
    _mock_soniox_module(monkeypatch)
    user_config = _build_user_config(
        language_hints=["not-a-language"],
        language_hints_strict=True,
    )
    audio_config = SimpleNamespace(transport_in_sample_rate=16000)

    service = create_stt_service(user_config, audio_config)

    assert service.params.language_hints is None
    assert service.params.language_hints_strict is None
