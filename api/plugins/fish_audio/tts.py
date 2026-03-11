import re

from loguru import logger
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

NON_SPEAKABLE_PATTERN = re.compile(r"^[\s\.,!?;:…\-–—()]+$")


def create_fish_tts(user_config, audio_config):
    """Create a Fish Audio TTS service with transport-aware sample-rate defaults."""
    try:
        from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
        from pipecat.services.fish.tts import FishAudioTTSService, ormsgpack
    except Exception as exc:
        logger.error(
            "Fish Audio SDK deps not found. Install pipecat with the fish extra: "
            "pip install 'pipecat-ai[fish]'"
        )
        logger.debug(f"Fish Audio import failure: {exc}")
        return None

    xml_function_tag_filter = XMLFunctionTagFilter()

    requested_rate = (
        getattr(
            user_config.tts,
            "telephony_sample_rate",
            audio_config.transport_out_sample_rate,
        )
        if getattr(audio_config, "transport_type", None)
        in {"twilio", "vonage", "vobiz", "cloudonix", "ari"}
        or audio_config.transport_out_sample_rate <= 8000
        else getattr(
            user_config.tts,
            "web_sample_rate",
            audio_config.transport_out_sample_rate,
        )
    )
    sample_rate = int(audio_config.transport_out_sample_rate)
    if requested_rate and int(requested_rate) != sample_rate:
        logger.warning(
            "Fish Audio requested sample_rate={} but transport requires {}. "
            "Using transport rate for compatibility.",
            requested_rate,
            sample_rate,
        )

    params = FishAudioTTSService.InputParams(
        latency=getattr(user_config.tts, "latency", "balanced") or "balanced",
        normalize=bool(getattr(user_config.tts, "normalize", True)),
        prosody_speed=float(getattr(user_config.tts, "speed", 1.0) or 1.0),
        prosody_volume=int(getattr(user_config.tts, "volume", 0) or 0),
    )

    class DograhFishTTSService(FishAudioTTSService):
        async def _receive_messages(self):
            async for message in self._get_websocket():
                try:
                    if not isinstance(message, bytes):
                        continue

                    payload = ormsgpack.unpackb(message)
                    if not isinstance(payload, dict):
                        continue

                    event = payload.get("event")
                    if event == "audio":
                        audio_data = payload.get("audio")
                        if audio_data:
                            await self.push_frame(
                                TTSAudioRawFrame(audio_data, self.sample_rate, 1)
                            )
                            await self.stop_ttfb_metrics()
                    elif event == "finish" and payload.get("reason") == "error":
                        await self.push_frame(
                            ErrorFrame(error="Fish Audio returned finish=error")
                        )
                except Exception as exc:
                    await self.push_error(
                        error_msg=f"Unknown error occurred: {exc}",
                        exception=exc,
                    )

        async def run_tts(self, text: str, context_id: str):
            normalized = (text or "").strip()
            if not normalized or NON_SPEAKABLE_PATTERN.fullmatch(normalized):
                logger.debug("Skipping non-speakable Fish TTS chunk: %r", text)
                return

            async for frame in super().run_tts(text, context_id):
                if frame is not None:
                    yield frame

    service = DograhFishTTSService(
        api_key=user_config.tts.api_key,
        reference_id=user_config.tts.voice,
        model_id=user_config.tts.model,
        output_format="pcm",
        sample_rate=sample_rate,
        params=params,
        append_trailing_space=True,
        text_filters=[xml_function_tag_filter],
    )
    # Our pipeline sends discrete turn-level utterances, not one continuous token stream.
    # Disable cross-chunk acoustic conditioning to avoid leaked prosody/content between turns.
    service._settings["condition_on_previous_chunks"] = False
    service._settings["top_p"] = float(getattr(user_config.tts, "top_p", 0.7) or 0.7)
    service._settings["temperature"] = float(
        getattr(user_config.tts, "temperature", 0.7) or 0.7
    )
    service._settings["chunk_length"] = int(
        getattr(user_config.tts, "chunk_length", 200) or 200
    )
    return service
