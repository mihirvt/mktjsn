from typing import TYPE_CHECKING

from fastapi import HTTPException
from loguru import logger

from api.constants import MPS_API_URL
from api.services.configuration.registry import ServiceProviders
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.dograh.llm import DograhLLMService
from pipecat.services.dograh.stt import DograhSTTService
from pipecat.services.dograh.tts import DograhTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.fireworks.llm import FireworksLLMService
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
from api.plugins.voicemaker import VoicemakerTTSService
from api.plugins.murf import MurfTTSService
from api.plugins.grok_tts import GrokTTSService
from api.plugins.inworld_tts import InworldTTSService
from api.plugins.cartesia.tts import create_cartesia_tts
from api.plugins.cartesia.stt import create_cartesia_stt
from api.plugins.fish_audio.tts import create_fish_tts
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

if TYPE_CHECKING:
    from api.services.pipecat.audio_config import AudioConfig


def _parse_soniox_language_hints(raw_hints) -> list[Language]:
    """Parse user-configured Soniox language hints into Pipecat Language enums."""
    if raw_hints is None:
        return []

    if isinstance(raw_hints, str):
        candidates = [raw_hints]
    elif isinstance(raw_hints, list):
        candidates = raw_hints
    else:
        logger.warning(f"Unsupported Soniox language_hints type: {type(raw_hints)}")
        return []

    parsed_hints: list[Language] = []
    for value in candidates:
        if not isinstance(value, str):
            continue
        normalized_value = value.strip()
        if not normalized_value:
            continue
        try:
            parsed_hints.append(Language(normalized_value))
        except ValueError:
            language_match = next(
                (
                    language
                    for language in Language
                    if str(language.value).lower() == normalized_value.lower()
                ),
                None,
            )
            if language_match:
                parsed_hints.append(language_match)
                continue

            logger.warning(f"Ignoring unsupported Soniox language hint: {value}")

    return parsed_hints


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
        try:
            service = create_cartesia_stt(user_config, audio_config)
            if service:
                return service
            
            # Fallback to standard Pipecat Cartesia service if plugin logic failed
            from pipecat.services.cartesia.stt import CartesiaLiveOptions, CartesiaSTTService
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
        except ImportError:
            logger.error("Cartesia SDK not found. Call will fail. Please install: pip install 'pipecat-ai[cartesia]'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Cartesia STT: {e}")
            raise
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
        mode = getattr(user_config.stt, "mode", None)
        params = SarvamSTTService.InputParams(language=pipecat_language)
        if mode:
            params.mode = mode

        return SarvamSTTService(
            api_key=user_config.stt.api_key,
            model=user_config.stt.model,
            params=params,
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
    elif user_config.stt.provider == ServiceProviders.SONIOX.value:
        from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService

        language_hints = _parse_soniox_language_hints(
            getattr(user_config.stt, "language_hints", None)
        )
        language_hints_strict = getattr(user_config.stt, "language_hints_strict", None)

        # Backward compatibility: if a legacy single `language` exists, treat it as one hint.
        if not language_hints:
            language_hints = _parse_soniox_language_hints(
                getattr(user_config.stt, "language", None)
            )

        params = SonioxInputParams(
            model=user_config.stt.model,
            language_hints=language_hints or None,
            language_hints_strict=(
                bool(language_hints_strict)
                if language_hints and language_hints_strict is not None
                else None
            ),
        )

        return SonioxSTTService(
            api_key=user_config.stt.api_key,
            params=params,
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
    elif user_config.tts.provider == ServiceProviders.FISH.value:
        service = create_fish_tts(user_config, audio_config)
        if service:
            return service
        raise HTTPException(
            status_code=500,
            detail="Fish Audio dependencies are not installed on the server",
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
    elif user_config.tts.provider == ServiceProviders.VOICEMAKER.value:
        voice = getattr(user_config.tts, "voice", None) or "ai3-Jony"
        is_telephony = audio_config.transport_out_sample_rate <= 8000
        if is_telephony:
            sample_rate = int(getattr(user_config.tts, "telephony_sample_rate", 8000) or 8000)
        else:
            sample_rate = int(getattr(user_config.tts, "web_sample_rate", 48000) or 48000)
        language_code = getattr(user_config.tts, "language", "en-US") or "en-US"
        # master_speed/pitch/volume are stored as float, API wants integer strings
        master_speed = str(int(float(getattr(user_config.tts, "master_speed", 0) or 0)))
        master_pitch = str(int(float(getattr(user_config.tts, "master_pitch", 0) or 0)))
        master_volume = str(int(float(getattr(user_config.tts, "master_volume", 0) or 0)))
        stability = getattr(user_config.tts, "stability", None)
        similarity = getattr(user_config.tts, "similarity", None)
        pro_engine = getattr(user_config.tts, "pro_engine", None)
        accent_code = getattr(user_config.tts, "accent_code", None)
        logger.debug(
            f"Voicemaker TTS: voice={voice}, language={language_code}, "
            f"sample_rate={sample_rate}, pro_engine={pro_engine}"
        )
        return VoicemakerTTSService(
            api_key=user_config.tts.api_key,
            voice_id=voice,
            sample_rate=sample_rate,
            params=VoicemakerTTSService.InputParams(
                language_code=language_code,
                output_format="mp3",  # We decode to PCM – always request mp3
                master_speed=master_speed,
                master_pitch=master_pitch,
                master_volume=master_volume,
                # stability/similarity stored as float, API needs integer string "50"
                stability=str(int(stability)) if stability is not None else None,
                similarity=str(int(similarity)) if similarity is not None else None,
                pro_engine=pro_engine if pro_engine else None,
                accent_code=accent_code if accent_code else None,
            ),
        )
    elif user_config.tts.provider == ServiceProviders.MURF.value:
        is_telephony = getattr(audio_config, "transport_type", None) in {
            "twilio",
            "vonage",
            "vobiz",
            "cloudonix",
            "ari",
        } or audio_config.transport_out_sample_rate <= 8000
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
        sample_rate = min(sample_rate, audio_config.transport_out_sample_rate)
        return MurfTTSService(
            api_key=user_config.tts.api_key,
            model=getattr(user_config.tts, "model", "FALCON") or "FALCON",
            sample_rate=sample_rate,
            params=MurfTTSService.InputParams(
                voice=getattr(user_config.tts, "voice", "Matthew") or "Matthew",
                locale=getattr(user_config.tts, "locale", "en-US") or "en-US",
                style=getattr(user_config.tts, "style", "Conversation")
                or "Conversation",
                rate=int(getattr(user_config.tts, "rate", 0) or 0),
                pitch=int(getattr(user_config.tts, "pitch", 0) or 0),
                min_buffer_size=int(getattr(user_config.tts, "min_buffer_size", 40) or 40),
                max_buffer_delay_in_ms=int(
                    getattr(user_config.tts, "max_buffer_delay_in_ms", 300) or 300
                ),
                region=getattr(user_config.tts, "region", "us-east") or "us-east",
            ),
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.GROK.value:
        is_telephony = getattr(audio_config, "transport_type", None) in {
            "twilio",
            "vonage",
            "vobiz",
            "cloudonix",
            "ari",
        } or audio_config.transport_out_sample_rate <= 8000
        if is_telephony:
            sample_rate = int(
                getattr(
                    user_config.tts,
                    "telephony_sample_rate",
                    audio_config.transport_out_sample_rate,
                )
                or audio_config.transport_out_sample_rate
            )
            # Cap telephony at 16 kHz
            sample_rate = min(sample_rate, 16000)
        else:
            sample_rate = int(
                getattr(
                    user_config.tts,
                    "web_sample_rate",
                    audio_config.transport_out_sample_rate,
                )
                or audio_config.transport_out_sample_rate
            )
        sample_rate = min(sample_rate, audio_config.transport_out_sample_rate)
        voice = getattr(user_config.tts, "voice", "eve") or "eve"
        language = getattr(user_config.tts, "language", "en") or "en"
        codec = getattr(user_config.tts, "codec", "pcm") or "pcm"
        return GrokTTSService(
            api_key=user_config.tts.api_key,
            voice=voice,
            codec=codec,
            sample_rate=sample_rate,
            language=language,
            params=GrokTTSService.InputParams(
                language=language,
            ),
            text_filters=[xml_function_tag_filter],
        )
    elif user_config.tts.provider == ServiceProviders.INWORLD.value:
        is_telephony = getattr(audio_config, "transport_type", None) in {
            "twilio",
            "vonage",
            "vobiz",
            "cloudonix",
            "ari",
        } or audio_config.transport_out_sample_rate <= 8000
        if is_telephony:
            sample_rate = int(
                getattr(
                    user_config.tts,
                    "telephony_sample_rate",
                    audio_config.transport_out_sample_rate,
                )
                or audio_config.transport_out_sample_rate
            )
            # Cap telephony at 16 kHz
            sample_rate = min(sample_rate, 16000)
        else:
            # For web calls, request from Inworld at whatever the user configured
            # or the plugin default (24kHz). Inworld natively produces best quality
            # at 24-48kHz. Capping it to pipeline 16kHz forces lossy server-side
            # downsampling which degrades clarity. The pipeline resampler will
            # handle conversion to transport_out rate (16kHz) far more cleanly.
            sample_rate = int(
                getattr(
                    user_config.tts,
                    "web_sample_rate",
                    24000,
                )
                or 24000
            )
        voice = getattr(user_config.tts, "voice", "Dennis") or "Dennis"
        model_id = getattr(user_config.tts, "model", "inworld-tts-1.5-max") or "inworld-tts-1.5-max"
        audio_encoding = getattr(user_config.tts, "audio_encoding", "PCM") or "PCM"
        # Auto-select MULAW only for Vonage (which uses mulaw natively).
        # Vobiz uses audio/x-l16 (raw PCM) — do NOT override to MULAW for vobiz.
        if is_telephony and audio_encoding == "PCM":
            transport_type = getattr(audio_config, "transport_type", None)
            if transport_type == "vonage":
                audio_encoding = "MULAW"
                logger.debug("Inworld TTS: auto-selecting MULAW for Vonage telephony")
        return InworldTTSService(
            api_key=user_config.tts.api_key,
            voice_id=voice,
            model_id=model_id,
            audio_encoding=audio_encoding,
            sample_rate=sample_rate,
            params=InworldTTSService.InputParams(
                temperature=float(getattr(user_config.tts, "temperature", 1.1) or 1.1),
                speaking_rate=float(getattr(user_config.tts, "speaking_rate", 1.0) or 1.0),
                auto_mode=bool(getattr(user_config.tts, "auto_mode", True)),
                buffer_char_threshold=int(
                    getattr(user_config.tts, "buffer_char_threshold", 60) or 60
                ),
                max_buffer_delay_ms=int(
                    getattr(user_config.tts, "max_buffer_delay_ms", 0) or 0
                ),
            ),
            text_filters=[xml_function_tag_filter],
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
        logger.debug(f"Creating Groq LLM service for model: {model}")
        temperature = getattr(user_config.llm, "temperature", 0.6)
        if temperature is None:
            temperature = 0.6
        service = GroqLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            params=OpenAILLMService.InputParams(temperature=temperature),
        )
        return service
    elif user_config.llm.provider == ServiceProviders.FIREWORKS.value:
        temperature = getattr(user_config.llm, "temperature", 0.1)
        if temperature is None:
            temperature = 0.1
        return FireworksLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            base_url=getattr(
                user_config.llm, "base_url", "https://api.fireworks.ai/inference/v1"
            ),
            params=OpenAILLMService.InputParams(temperature=temperature),
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
        service = DeepInfraLLMService(
            api_key=user_config.llm.api_key,
            model=model,
            params=OpenAILLMService.InputParams(
                temperature=temperature,
                extra={"reasoning_effort": reasoning_effort},
            ),
        )
        return service
    else:
        raise HTTPException(status_code=400, detail="Invalid LLM provider")
