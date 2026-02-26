from typing import TYPE_CHECKING

from fastapi import HTTPException
from loguru import logger

from api.constants import MPS_API_URL
from api.services.configuration.registry import ServiceProviders
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.cartesia.stt import CartesiaLiveOptions, CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.dograh.llm import DograhLLMService
from pipecat.services.dograh.stt import DograhSTTService
from pipecat.services.dograh.tts import DograhTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.sarvam.tts import SarvamTTSService
from api.plugins.sarvam import SarvamLLMService
from api.plugins.deepinfra import DeepInfraLLMService
from api.plugins.smallest_ai import SmallestAITTSService
from api.plugins.cartesia.tts import create_cartesia_tts
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

if TYPE_CHECKING:
    from api.services.pipecat.audio_config import AudioConfig


def create_stt_service(
    user_config, audio_config: "AudioConfig", keyterms: list[str] | None = None
):
    """Create and return appropriate STT service based on user configuration

    Args:
        user_config: User configuration containing STT settings
        keyterms: Optional list of keyterms for speech recognition boosting (Deepgram only)
    """
    logger.info(
        f"Creating STT service: provider={user_config.stt.provider}, model={user_config.stt.model}"
    )
    if user_config.stt.provider == ServiceProviders.DEEPGRAM.value:
        # Check if using Flux model (English-only, no language selection)
        if user_config.stt.model == "flux-general-en":
            logger.debug("Using DeepGram Flux Model")
            return DeepgramFluxSTTService(
                api_key=user_config.stt.api_key,
                model=user_config.stt.model,
                params=DeepgramFluxSTTService.InputParams(
                    eot_timeout_ms=3000,
                    eot_threshold=0.7,
                    keyterm=keyterms or [],
                ),
                should_interrupt=False,  # Let UserAggregator take care of sending InterruptionFrame
                sample_rate=audio_config.transport_in_sample_rate,
            )

        # Other models than flux
        # Use language from user config, defaulting to "multi" for multilingual support
        language = getattr(user_config.stt, "language", None) or "multi"
        live_options = LiveOptions(
            language=language,
            profanity_filter=False,
            endpointing=100,
            model=user_config.stt.model,
            keyterm=keyterms or [],
        )
        logger.debug(f"Using DeepGram Model - {user_config.stt.model}")
        return DeepgramSTTService(
            live_options=live_options,
            api_key=user_config.stt.api_key,
            should_interrupt=False,  # Let UserAggregator take care of sending InterruptionFrame
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.OPENAI.value:
        return OpenAISTTService(
            api_key=user_config.stt.api_key, model=user_config.stt.model
        )
    elif user_config.stt.provider == ServiceProviders.CARTESIA.value:
        language = getattr(user_config.stt, "language", "hi") or "hi"
        return CartesiaSTTService(
            api_key=user_config.stt.api_key,
            live_options=CartesiaLiveOptions(
                model=user_config.stt.model,
                language=language,
                encoding="pcm_s16le",
                sample_rate=audio_config.transport_in_sample_rate,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.DOGRAH.value:
        base_url = MPS_API_URL.replace("http://", "ws://").replace("https://", "wss://")
        language = getattr(user_config.stt, "language", None) or "multi"
        return DograhSTTService(
            base_url=base_url,
            api_key=user_config.stt.api_key,
            model=user_config.stt.model,
            language=language,
            keyterms=keyterms,
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SARVAM.value:
        # Map Sarvam language code to pipecat Language enum
        language_mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
        }
        language = getattr(user_config.stt, "language", None)
        pipecat_language = language_mapping.get(language, Language.HI_IN)
        return SarvamSTTService(
            api_key=user_config.stt.api_key,
            model=user_config.stt.model,
            params=SarvamSTTService.InputParams(language=pipecat_language),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SPEECHMATICS.value:
        from pipecat.services.speechmatics.stt import (
            AdditionalVocabEntry,
            OperatingPoint,
        )

        language = getattr(user_config.stt, "language", None) or "en"
        # Map model field to operating point (standard or enhanced)
        operating_point = (
            OperatingPoint.ENHANCED
            if user_config.stt.model == "enhanced"
            else OperatingPoint.STANDARD
        )
        # Convert keyterms to AdditionalVocabEntry objects for Speechmatics
        additional_vocab = []
        if keyterms:
            additional_vocab = [AdditionalVocabEntry(content=term) for term in keyterms]
        return SpeechmaticsSTTService(
            api_key=user_config.stt.api_key,
            params=SpeechmaticsSTTService.InputParams(
                language=language,
                operating_point=operating_point,
                additional_vocab=additional_vocab,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid STT provider {user_config.stt.provider}"
        )


def create_tts_service(user_config, audio_config: "AudioConfig"):
    """Create and return appropriate TTS service based on user configuration

    Args:
        user_config: User configuration containing TTS settings
        transport_type: Type of transport (e.g., 'twilio', 'webrtc')
    """
    logger.info(
        f"Creating TTS service: provider={user_config.tts.provider}, model={user_config.tts.model}"
    )
    # Create function call filter to prevent TTS from speaking function call tags
    xml_function_tag_filter = XMLFunctionTagFilter()
    if user_config.tts.provider == ServiceProviders.DEEPGRAM.value:
        return DeepgramTTSService(
            api_key=user_config.tts.api_key,
            voice=user_config.tts.voice,
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.OPENAI.value:
        return OpenAITTSService(
            api_key=user_config.tts.api_key,
            model=user_config.tts.model,
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.ELEVENLABS.value:
        # Backward compatible with older configuration "Name - voice_id"
        try:
            voice_id = user_config.tts.voice.split(" - ")[1]
        except IndexError:
            voice_id = user_config.tts.voice
        return ElevenLabsTTSService(
            reconnect_on_error=False,
            api_key=user_config.tts.api_key,
            voice_id=voice_id,
            model=user_config.tts.model,
            params=ElevenLabsTTSService.InputParams(
                stability=0.8, speed=user_config.tts.speed, similarity_boost=0.75
            ),
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.CARTESIA.value:
        try:
            service = create_cartesia_tts(user_config, audio_config)
            if service:
                return service
            
            # Fallback to standard Pipecat Cartesia service if plugin logic returned None
            from pipecat.services.cartesia.tts import CartesiaTTSService
            return CartesiaTTSService(
                api_key=user_config.tts.api_key,
                voice_id=user_config.tts.voice,
                model=user_config.tts.model,
                text_filters=[xml_function_tag_filter],
            )
        except ImportError:
            logger.error("Cartesia SDK not found. Call will fail. Please install: pip install 'pipecat-ai[cartesia]'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Cartesia TTS: {e}")
            raise
    elif user_config.tts.provider == ServiceProviders.DOGRAH.value:
        # Convert HTTP URL to WebSocket URL for TTS
        base_url = MPS_API_URL.replace("http://", "ws://").replace("https://", "wss://")
        return DograhTTSService(
            base_url=base_url,
            api_key=user_config.tts.api_key,
            model=user_config.tts.model,
            voice=user_config.tts.voice,
            params=DograhTTSService.InputParams(speed=user_config.tts.speed),
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.SARVAM.value:
        # Map Sarvam language code to pipecat Language enum for TTS
        language_mapping = {
            "bn-IN": Language.BN,
            "en-IN": Language.EN,
            "gu-IN": Language.GU,
            "hi-IN": Language.HI,
            "kn-IN": Language.KN,
            "ml-IN": Language.ML,
            "mr-IN": Language.MR,
            "od-IN": Language.OR,
            "pa-IN": Language.PA,
            "ta-IN": Language.TA,
            "te-IN": Language.TE,
        }
        language = getattr(user_config.tts, "language", None)
        pipecat_language = language_mapping.get(language, Language.HI)

        voice = getattr(user_config.tts, "voice", None) or "anushka"
        return SarvamTTSService(
            api_key=user_config.tts.api_key,
            model=user_config.tts.model,
            voice_id=voice,
            params=SarvamTTSService.InputParams(language=pipecat_language),
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.SMALLEST_AI.value:
        voice = getattr(user_config.tts, "voice", None) or "ryan"
        # Use per-transport sample rate from user config, falling back to audio_config
        is_telephony = audio_config.transport_out_sample_rate <= 8000
        if is_telephony:
            sample_rate = int(getattr(user_config.tts, "telephony_sample_rate", 8000) or 8000)
        else:
            sample_rate = int(getattr(user_config.tts, "web_sample_rate", 16000) or 16000)
        return SmallestAITTSService(
            api_key=user_config.tts.api_key,
            model=user_config.tts.model,
            voice_id=voice,
            sample_rate=sample_rate,
            params=SmallestAITTSService.InputParams(
                language=user_config.tts.language,
                speed=user_config.tts.speed,
                max_buffer_flush_ms=user_config.tts.max_buffer_flush_ms,
                consistency=float(getattr(user_config.tts, "consistency", 0.5)),
                enhancement=int(getattr(user_config.tts, "enhancement", 1)),
                similarity=float(getattr(user_config.tts, "similarity", 0)),
            ),
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid TTS provider {user_config.tts.provider}"
        )


def create_llm_service(user_config):
    """Create and return appropriate LLM service based on user configuration"""
    model = user_config.llm.model
    logger.info(
        f"Creating LLM service: provider={user_config.llm.provider}, model={model}"
    )
    if user_config.llm.provider == ServiceProviders.OPENAI.value:
        if "gpt-5" in model:
            return OpenAILLMService(
                api_key=user_config.llm.api_key,
                model=model,
                params=OpenAILLMService.InputParams(
                    reasoning_effort="minimal", verbosity="low"
                ),
            )
        else:
            return OpenAILLMService(
                api_key=user_config.llm.api_key,
                model=model,
                params=OpenAILLMService.InputParams(temperature=0.1),
            )
    elif user_config.llm.provider == ServiceProviders.GROQ.value:
        print(
            f"Creating Groq LLM service with API key: {user_config.llm.api_key} and model: {model}"
        )
        return GroqLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            params=OpenAILLMService.InputParams(temperature=0.1),
        )
    elif user_config.llm.provider == ServiceProviders.OPENROUTER.value:
        return OpenRouterLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            base_url=user_config.llm.base_url,
            params=OpenAILLMService.InputParams(temperature=0.1),
        )
    elif user_config.llm.provider == ServiceProviders.GOOGLE.value:
        # Use the correct InputParams class for Google to avoid propagating OpenAI-specific
        # NOT_GIVEN sentinels that break Pydantic validation in GoogleLLMService.
        temperature = getattr(user_config.llm, "temperature", 0.1) or 0.1
        return GoogleLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            params=GoogleLLMService.InputParams(temperature=temperature),
        )
    elif user_config.llm.provider == ServiceProviders.AZURE.value:
        return AzureLLMService(
            api_key=user_config.llm.api_key,
            endpoint=user_config.llm.endpoint,
            model=model,  # Azure uses deployment name as model
            params=AzureLLMService.InputParams(temperature=0.1),
        )
    elif user_config.llm.provider == ServiceProviders.DOGRAH.value:
        return DograhLLMService(
            base_url=f"{MPS_API_URL}/api/v1/llm",
            api_key=user_config.llm.api_key,
            model=model,
        )
    elif user_config.llm.provider == ServiceProviders.SARVAM.value:
        return SarvamLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            params=OpenAILLMService.InputParams(temperature=0.1),
        )
    elif user_config.llm.provider == ServiceProviders.DEEPINFRA.value:
        reasoning_effort = getattr(user_config.llm, "reasoning_effort", "none") or "none"
        temperature = getattr(user_config.llm, "temperature", 0.1) or 0.1
        return DeepInfraLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            params=OpenAILLMService.InputParams(
                temperature=temperature,
                extra={"reasoning_effort": reasoning_effort},
            ),
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid LLM provider")
