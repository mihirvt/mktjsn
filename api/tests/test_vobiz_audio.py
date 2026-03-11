import base64
import json

import pytest

from api.services.pipecat.audio_config import create_vobiz_audio_config
from pipecat.frames.frames import AudioRawFrame
from pipecat.serializers.vobiz import VobizFrameSerializer


def test_create_vobiz_audio_config_for_l16_16khz():
    audio_config = create_vobiz_audio_config("audio/x-l16", 16000)

    assert audio_config.transport_type == "vobiz"
    assert audio_config.transport_in_sample_rate == 16000
    assert audio_config.transport_out_sample_rate == 16000
    assert audio_config.pipeline_sample_rate == 16000
    assert audio_config.vad_sample_rate == 16000


@pytest.mark.asyncio
async def test_vobiz_serializer_outputs_l16_playaudio():
    serializer = VobizFrameSerializer(
        stream_id="stream-1",
        call_id="call-1",
        auth_id="auth-id",
        auth_token="auth-token",
        params=VobizFrameSerializer.InputParams(
            vobiz_sample_rate=16000,
            vobiz_content_type="audio/x-l16",
            sample_rate=16000,
        ),
    )

    async def _passthrough(data, in_rate, out_rate):
        assert in_rate == 16000
        assert out_rate == 16000
        return data

    serializer._output_resampler.resample = _passthrough  # type: ignore[method-assign]

    frame = AudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1)
    serialized = await serializer.serialize(frame)
    payload = json.loads(serialized)

    assert payload["event"] == "playAudio"
    assert payload["media"]["contentType"] == "audio/x-l16"
    assert payload["media"]["sampleRate"] == 16000
    assert base64.b64decode(payload["media"]["payload"]) == b"\x01\x02\x03\x04"
