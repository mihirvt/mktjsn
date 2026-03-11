from loguru import logger
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter


def create_fish_tts(user_config, audio_config):
    """Create a Fish Audio TTS service with transport-aware sample-rate defaults."""
    try:
        from pipecat.services.fish.tts import FishAudioTTSService
    except Exception as exc:
        logger.error(
            "Fish Audio SDK deps not found. Install pipecat with the fish extra: "
            "pip install 'pipecat-ai[fish]'"
        )
        logger.debug(f"Fish Audio import failure: {exc}")
        return None

    xml_function_tag_filter = XMLFunctionTagFilter()

    is_telephony = (
        getattr(audio_config, "transport_type", None)
        in {"twilio", "vonage", "vobiz", "cloudonix", "ari"}
        or audio_config.transport_out_sample_rate <= 8000
    )
    if is_telephony:
        sample_rate = int(
            getattr(
                user_config.tts,
                "telephony_sample_rate",
                audio_config.transport_out_sample_rate,
            )
            or audio_config.transport_out_sample_rate
        )
    else:
        sample_rate = int(
            getattr(
                user_config.tts,
                "web_sample_rate",
                audio_config.transport_out_sample_rate,
            )
            or audio_config.transport_out_sample_rate
        )

    params = FishAudioTTSService.InputParams(
        latency=getattr(user_config.tts, "latency", "balanced") or "balanced",
        normalize=bool(getattr(user_config.tts, "normalize", True)),
        prosody_speed=float(getattr(user_config.tts, "speed", 1.0) or 1.0),
        prosody_volume=int(getattr(user_config.tts, "volume", 0) or 0),
    )

    return FishAudioTTSService(
        api_key=user_config.tts.api_key,
        reference_id=user_config.tts.voice,
        model_id=user_config.tts.model,
        output_format="pcm",
        sample_rate=sample_rate,
        params=params,
        text_filters=[xml_function_tag_filter],
    )
