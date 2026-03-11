import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from api.plugins.murf import MurfTTSService
from api.routes.user import get_voices
from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import create_tts_service
from api.services.pricing.tts import TTS_PRICING
from pipecat.frames.frames import TTSAudioRawFrame


@pytest.mark.asyncio
async def test_murf_audio_chunks_strip_wav_header_once():
    service = MurfTTSService(api_key="test-key", sample_rate=16000)
    pushed_frames = []

    async def _capture(frame, *args, **kwargs):
        pushed_frames.append(frame)

    service.push_frame = _capture  # type: ignore[method-assign]
    service._active_context_id = "ctx-1"
    service._context_finished.clear()
    service._header_pending_contexts.add("ctx-1")

    wav_header = bytes(range(44))
    pcm_payload = b"\x01\x02\x03\x04"
    await service._handle_text_message(
        json.dumps(
            {
                "context_id": "ctx-1",
                "audio": base64.b64encode(wav_header + pcm_payload).decode("utf-8"),
            }
        )
    )
    await service._handle_text_message(
        json.dumps(
            {
                "context_id": "ctx-1",
                "audio": base64.b64encode(pcm_payload).decode("utf-8"),
            }
        )
    )
    await service._handle_text_message(json.dumps({"context_id": "ctx-1", "final": True}))

    assert len(pushed_frames) == 2
    assert all(isinstance(frame, TTSAudioRawFrame) for frame in pushed_frames)
    assert pushed_frames[0].audio == pcm_payload
    assert pushed_frames[1].audio == pcm_payload
    assert service._context_finished.is_set()


@pytest.mark.asyncio
async def test_murf_drops_stale_audio_after_clear():
    service = MurfTTSService(api_key="test-key", sample_rate=16000)
    pushed_frames = []

    async def _capture(frame, *args, **kwargs):
        pushed_frames.append(frame)

    async def _noop_send_json(_payload):
        return None

    service.push_frame = _capture  # type: ignore[method-assign]
    service._send_json = _noop_send_json  # type: ignore[method-assign]
    service._active_context_id = "ctx-stale"
    await service._clear_active_context()

    await service._handle_text_message(
        json.dumps(
            {
                "context_id": "ctx-stale",
                "audio": base64.b64encode(b"\x00" * 48).decode("utf-8"),
            }
        )
    )

    assert pushed_frames == []
    assert service._context_finished.is_set()


@pytest.mark.asyncio
async def test_murf_sends_advanced_settings_once_per_connection():
    sent_payloads = []
    service = MurfTTSService(
        api_key="test-key",
        sample_rate=16000,
        params=MurfTTSService.InputParams(
            min_buffer_size=60,
            max_buffer_delay_in_ms=500,
        ),
    )

    async def _capture_send(payload):
        sent_payloads.append(payload)

    async def _noop_connect():
        return None

    service._send_json = _capture_send  # type: ignore[method-assign]
    service._connect = _noop_connect  # type: ignore[method-assign]
    service._ws = SimpleNamespace(closed=False)

    async for _ in service.run_tts("Hello there.", context_id="ctx-1"):
        pass
    service._context_finished.set()
    async for _ in service.run_tts("Second turn.", context_id="ctx-2"):
        pass

    assert sent_payloads[0] == {
        "min_buffer_size": 60,
        "max_buffer_delay_in_ms": 500,
    }
    assert sent_payloads[1]["context_id"] == "ctx-1"
    assert sent_payloads[2]["context_id"] == "ctx-1"
    assert sent_payloads[3]["context_id"] == "ctx-2"
    assert sent_payloads[4]["context_id"] == "ctx-2"


def test_create_tts_service_returns_murf_and_transport_aware_sample_rate():
    user_config = SimpleNamespace(
        tts=SimpleNamespace(
            provider=ServiceProviders.MURF.value,
            api_key="test-key",
            model="FALCON",
            voice="Matthew",
            locale="en-US",
            style="Conversation",
            rate=0,
            pitch=0,
            min_buffer_size=40,
            max_buffer_delay_in_ms=300,
            region="us-east",
            telephony_sample_rate=8000,
            web_sample_rate=16000,
        )
    )

    telephony_audio = AudioConfig(
        transport_in_sample_rate=8000,
        transport_out_sample_rate=8000,
        vad_sample_rate=8000,
        pipeline_sample_rate=8000,
    )
    web_audio = AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
        vad_sample_rate=16000,
        pipeline_sample_rate=16000,
    )

    telephony_service = create_tts_service(user_config, telephony_audio)
    web_service = create_tts_service(user_config, web_audio)

    assert isinstance(telephony_service, MurfTTSService)
    assert isinstance(web_service, MurfTTSService)
    assert telephony_service._sample_rate == 8000
    assert web_service._sample_rate == 16000
    assert telephony_service._params.min_buffer_size == 40
    assert telephony_service._params.max_buffer_delay_in_ms == 300


def test_murf_pricing_registered():
    assert TTS_PRICING[ServiceProviders.MURF]["FALCON"].price_per_character > 0


@pytest.mark.asyncio
async def test_get_voices_murf_maps_supported_locales(monkeypatch):
    async def _mock_get_user_configurations(_user_id):
        return SimpleNamespace(
            tts=SimpleNamespace(
                provider=ServiceProviders.MURF.value,
                api_key="test-key",
            )
        )

    class MockResponse:
        status_code = 200

        def json(self):
            return [
                {
                    "voiceId": "Matthew",
                    "displayName": "Matthew",
                    "gender": "Male",
                    "locale": "en-US",
                    "supportedLocales": {
                        "en-US": {"availableStyles": ["Conversation"], "detail": "English - US & Canada"},
                        "fr-CA": {"availableStyles": ["Conversation"], "detail": "French - Canada"},
                    },
                }
            ]

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return MockResponse()

    monkeypatch.setattr("api.routes.user.db_client.get_user_configurations", _mock_get_user_configurations)
    monkeypatch.setattr("httpx.AsyncClient", MockAsyncClient)

    user = SimpleNamespace(id=1)
    response = await get_voices("murf", user)

    assert response.provider == "murf"
    assert len(response.voices) == 1
    assert response.voices[0].supported_locales == ["en-US", "fr-CA"]
    assert response.voices[0].styles_by_locale == {
        "en-US": ["Conversation"],
        "fr-CA": ["Conversation"],
    }
