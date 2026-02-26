from loguru import logger
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

# Move imports inside functions or wrap in try-except to avoid crashing the factory if SDK is missing
def create_cartesia_tts(user_config, audio_config):
    """
    Creates a Cartesia TTS service with custom parameters (emotions, sample rate).
    Follows MUKTAM modularity rule by isolating custom logic.
    """
    try:
        from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
    except ImportError:
        logger.error("Cartesia SDK not found. Please install with: pip install cartesia pipecat-ai[cartesia]")
        return None

    # Create function call filter
    xml_function_tag_filter = XMLFunctionTagFilter()

    # Match SmallestAI logic for sample rate selection
    # Telephony is usually <= 16000
    is_telephony = audio_config.transport_out_sample_rate <= 16000
    
    # We should generally match the pipeline rate to avoid issues with 
    # aggregators and buffers, unless the transport handles the mismatch perfectly.
    pipeline_rate = getattr(audio_config, "pipeline_sample_rate", 16000)
    
    if is_telephony:
        requested_rate = int(
            getattr(user_config.tts, "telephony_sample_rate", 16000) or 16000
        )
    else:
        requested_rate = int(
            getattr(user_config.tts, "web_sample_rate", 44100) or 44100
        )
    
    # Cap at pipeline rate to prevent disconnects from sample rate mismatch
    # if the pipeline is at 16k, pushing 44.1k frames will confuse buffers.
    sample_rate = min(requested_rate, pipeline_rate)
    
    if requested_rate > pipeline_rate:
        logger.warning(
            f"Cartesia requested {requested_rate}Hz but pipeline is {pipeline_rate}Hz. "
            f"Capping to {pipeline_rate}Hz for stability."
        )

    # Build Cartesia-specific parameters
    params = {}
    try:
        # Cartesia Sonic-3 supports generation_config for emotions
        params = CartesiaTTSService.InputParams(
            generation_config=GenerationConfig(
                emotion=getattr(user_config.tts, "emotion", "neutral")
            )
        )
    except Exception as e:
        logger.warning(f"Failed to use GenerationConfig for Cartesia: {e}. Falling back...")
        try:
            params = CartesiaTTSService.InputParams(
                emotion=getattr(user_config.tts, "emotion", "neutral")
            )
        except:
            params = None

    logger.debug(
        f"Cartesia TTS Plugin: sample_rate={sample_rate}, emotion={getattr(user_config.tts, 'emotion', 'neutral')}"
    )

    return CartesiaTTSService(
        api_key=user_config.tts.api_key,
        voice_id=user_config.tts.voice,
        model=user_config.tts.model,
        sample_rate=sample_rate,
        params=params,
        text_filters=[xml_function_tag_filter],
    )
