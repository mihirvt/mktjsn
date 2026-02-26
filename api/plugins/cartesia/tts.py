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
    
    # Do NOT cap at pipeline rate for Cartesia, as the web transport handles
    # the mismatch perfectly and downsizing later is supported.
    sample_rate = requested_rate
    
    if requested_rate > pipeline_rate:
        logger.debug(
            f"Cartesia requested {requested_rate}Hz but pipeline is {pipeline_rate}Hz. "
            f"Cartesia supports this natively for web calls."
        )

    # Build Cartesia-specific parameters
    params = {}
    cartesia_emotion = getattr(user_config.tts, "emotion", "neutral")
    cartesia_speed = getattr(user_config.tts, "speed", 1.0)
    cartesia_language = getattr(user_config.tts, "language", "en")
    
    try:
        # Cartesia Sonic-3 supports generation_config for emotions and speed
        gen_config = GenerationConfig(
            emotion=cartesia_emotion,
            speed=cartesia_speed
        )
        
        # Try to pass add_context / continue_ logic to avoid double-message interruption issues
        # Also request timestamps so pipecat knows exactly how much got spoken on interruptions
        try:
            params = CartesiaTTSService.InputParams(
                generation_config=gen_config,
                language=cartesia_language,
                add_timestamps=True,
                continue_=False  # Pipecat uses this to not hold context during interruptions
            )
        except Exception:
            params = CartesiaTTSService.InputParams(
                generation_config=gen_config,
                language=cartesia_language
            )
            
    except Exception as e:
        logger.warning(f"Failed to use full InputParams for Cartesia: {e}. Falling back...")
        try:
            params = CartesiaTTSService.InputParams(
                emotion=cartesia_emotion,
                speed=cartesia_speed,
                language=cartesia_language
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
