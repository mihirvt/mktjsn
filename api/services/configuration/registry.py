from enum import Enum, auto
from typing import Annotated, Dict, Literal, Type, TypeVar, Union

from pydantic import BaseModel, Field, computed_field


class ServiceType(Enum):
    LLM = auto()
    TTS = auto()
    STT = auto()
    EMBEDDINGS = auto()


class ServiceProviders(str, Enum):
    OPENAI = "openai"
    FIREWORKS = "fireworks"
    DEEPGRAM = "deepgram"
    FISH = "fish"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    CARTESIA = "cartesia"
    # NEUPHONIC = "neuphonic"
    ELEVENLABS = "elevenlabs"
    GOOGLE = "google"
    AZURE = "azure"
    DOGRAH = "dograh"
    SARVAM = "sarvam"
    SPEECHMATICS = "speechmatics"
    SMALLEST_AI = "smallest_ai"
    DEEPINFRA = "deepinfra"
    SONIOX = "soniox"
    VOICEMAKER = "voicemaker"
    MURF = "murf"


class BaseServiceConfiguration(BaseModel):
    provider: Literal[
        ServiceProviders.OPENAI,
        ServiceProviders.FIREWORKS,
        ServiceProviders.DEEPGRAM,
        ServiceProviders.FISH,
        ServiceProviders.GROQ,
        ServiceProviders.OPENROUTER,
        ServiceProviders.ELEVENLABS,
        ServiceProviders.GOOGLE,
        ServiceProviders.AZURE,
        ServiceProviders.DOGRAH,
        ServiceProviders.DEEPINFRA,
        # ServiceProviders.SARVAM,
    ]
    api_key: str


class BaseLLMConfiguration(BaseServiceConfiguration):
    model: str


class BaseTTSConfiguration(BaseServiceConfiguration):
    model: str


class BaseSTTConfiguration(BaseServiceConfiguration):
    model: str


class BaseEmbeddingsConfiguration(BaseServiceConfiguration):
    model: str


# Unified registry for all service types
REGISTRY: Dict[ServiceType, Dict[str, Type[BaseServiceConfiguration]]] = {
    ServiceType.LLM: {},
    ServiceType.TTS: {},
    ServiceType.STT: {},
    ServiceType.EMBEDDINGS: {},
}

T = TypeVar("T", bound=BaseServiceConfiguration)


def register_service(service_type: ServiceType):
    """Generic decorator for registering service configurations"""

    def decorator(cls: Type[T]) -> Type[T]:
        # Get provider from class attributes or field defaults
        provider = getattr(cls, "provider", None)
        if provider is None:
            # Try to get from model fields
            provider = cls.model_fields.get("provider", None)
            if provider is not None:
                provider = provider.default
        if provider is None:
            raise ValueError(f"Provider not specified for {cls.__name__}")

        REGISTRY[service_type][provider] = cls
        return cls

    return decorator


# Convenience decorators
def register_llm(cls: Type[BaseLLMConfiguration]):
    return register_service(ServiceType.LLM)(cls)


def register_tts(cls: Type[BaseTTSConfiguration]):
    return register_service(ServiceType.TTS)(cls)


def register_stt(cls: Type[BaseSTTConfiguration]):
    return register_service(ServiceType.STT)(cls)


def register_embeddings(cls: Type[BaseEmbeddingsConfiguration]):
    return register_service(ServiceType.EMBEDDINGS)(cls)


###################################################### LLM ########################################################################

# Suggested models for each provider (used for UI dropdown)
OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-3.5-turbo",
]
FIREWORKS_MODELS = [
    "accounts/fireworks/models/firefunction-v2",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
]
GOOGLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
]
OPENROUTER_MODELS = [
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat-v3-0324",
]
AZURE_MODELS = ["gpt-4.1-mini"]
DOGRAH_LLM_MODELS = ["default", "accurate", "fast", "lite", "zen"]


@register_llm
class OpenAILLMService(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.OPENAI] = ServiceProviders.OPENAI
    model: str = Field(default="gpt-4.1", json_schema_extra={"examples": OPENAI_MODELS})
    api_key: str


@register_llm
class FireworksLLMConfiguration(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.FIREWORKS] = ServiceProviders.FIREWORKS
    model: str = Field(
        default="accounts/fireworks/models/firefunction-v2",
        json_schema_extra={"examples": FIREWORKS_MODELS},
    )
    api_key: str
    base_url: str = Field(default="https://api.fireworks.ai/inference/v1")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0,
        description="Sampling temperature (0 = focused, 2 = creative)",
    )


@register_llm
class GoogleLLMService(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.GOOGLE] = ServiceProviders.GOOGLE
    model: str = Field(
        default="gemini-2.0-flash", json_schema_extra={"examples": GOOGLE_MODELS}
    )
    api_key: str
    temperature: Union[float, None] = Field(default=0.1)


@register_llm
class GroqLLMService(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.GROQ] = ServiceProviders.GROQ
    model: str = Field(
        default="llama-3.3-70b-versatile", json_schema_extra={"examples": GROQ_MODELS}
    )
    api_key: str
    temperature: float = Field(
        default=0.6, ge=0.0, le=2.0,
        description="Sampling temperature (0 = focused, 2 = creative)",
    )


@register_llm
class OpenRouterLLMConfiguration(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.OPENROUTER] = ServiceProviders.OPENROUTER
    model: str = Field(
        default="openai/gpt-4.1", json_schema_extra={"examples": OPENROUTER_MODELS}
    )
    api_key: str
    base_url: str = Field(default="https://openrouter.ai/api/v1")


@register_llm
class AzureLLMService(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.AZURE] = ServiceProviders.AZURE
    model: str = Field(
        default="gpt-4.1-mini", json_schema_extra={"examples": AZURE_MODELS}
    )
    api_key: str
    endpoint: str


@register_llm
class DograhLLMService(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.DOGRAH] = ServiceProviders.DOGRAH
    model: str = Field(
        default="default", json_schema_extra={"examples": DOGRAH_LLM_MODELS}
    )
    api_key: str


SARVAM_LLM_MODELS = ["sarvam-m", "sarvam-m-2"]


@register_llm
class SarvamLLMService(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.SARVAM] = ServiceProviders.SARVAM
    model: str = Field(
        default="sarvam-m", json_schema_extra={"examples": SARVAM_LLM_MODELS}
    )
    api_key: str


DEEPINFRA_LLM_MODELS = [
    "moonshotai/Kimi-K2.5",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B",
    "google/gemma-3-27b-it",
]
DEEPINFRA_REASONING_EFFORTS = ["none", "low", "medium", "high"]


@register_llm
class DeepInfraLLMConfiguration(BaseLLMConfiguration):
    provider: Literal[ServiceProviders.DEEPINFRA] = ServiceProviders.DEEPINFRA
    model: str = Field(
        default="moonshotai/Kimi-K2.5",
        json_schema_extra={"examples": DEEPINFRA_LLM_MODELS},
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0,
        description="Sampling temperature (0 = focused, 2 = creative)",
    )
    reasoning_effort: str = Field(
        default="none",
        json_schema_extra={"examples": DEEPINFRA_REASONING_EFFORTS},
    )
    api_key: str


LLMConfig = Annotated[
    Union[
        OpenAILLMService,
        FireworksLLMConfiguration,
        GroqLLMService,
        OpenRouterLLMConfiguration,
        GoogleLLMService,
        AzureLLMService,
        DograhLLMService,
        SarvamLLMService,
        DeepInfraLLMConfiguration,
    ],
    Field(discriminator="provider"),
]

###################################################### TTS ########################################################################


@register_tts
class DeepgramTTSConfiguration(BaseServiceConfiguration):
    provider: Literal[ServiceProviders.DEEPGRAM] = ServiceProviders.DEEPGRAM
    voice: str = "aura-2-helena-en"
    api_key: str

    @computed_field
    @property
    def model(self) -> str:
        # Deepgram model's name is inferred using the voice name.
        # It can either contain aura-2 or aura-1
        if "aura-2" in self.voice:
            return "aura-2"
        elif "aura-1" in self.voice:
            return "aura-1"
        else:
            # Default fallback
            return "aura-2"


ELEVENLABS_TTS_MODELS = ["eleven_flash_v2_5"]


@register_tts
class ElevenlabsTTSConfiguration(BaseServiceConfiguration):
    provider: Literal[ServiceProviders.ELEVENLABS] = ServiceProviders.ELEVENLABS
    voice: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID
    speed: float = Field(default=1.0, ge=0.1, le=2.0, description="Speed of the voice")
    model: str = Field(
        default="eleven_flash_v2_5",
        json_schema_extra={"examples": ELEVENLABS_TTS_MODELS},
    )
    api_key: str


FISH_TTS_MODELS = ["s2-pro", "s1"]
FISH_TTS_LATENCIES = ["balanced", "normal"]
FISH_TTS_SAMPLE_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


@register_tts
class FishTTSConfiguration(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.FISH] = ServiceProviders.FISH
    model: str = Field(
        default="s2-pro",
        json_schema_extra={"examples": FISH_TTS_MODELS},
        description="Fish live WebSocket model header",
    )
    voice: str = Field(
        default="bf322df2096a46f18c579d0baa36f41d",
        description="Fish voice reference_id",
    )
    latency: str = Field(
        default="balanced",
        json_schema_extra={"examples": FISH_TTS_LATENCIES},
        description="Balanced favors latency; normal favors quality",
    )
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    chunk_length: int = Field(default=200, ge=100, le=300)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: int = Field(default=0, ge=-20, le=20)
    normalize: bool = Field(
        default=True,
        description="Text normalization before synthesis",
    )
    telephony_sample_rate: int = Field(
        default=8000,
        json_schema_extra={"examples": FISH_TTS_SAMPLE_RATES},
        description="Preferred telephony sample rate; runtime output is clamped to the active transport format",
    )
    web_sample_rate: int = Field(
        default=16000,
        json_schema_extra={"examples": FISH_TTS_SAMPLE_RATES},
        description="Preferred web/WebRTC sample rate; runtime output is clamped to the active transport format",
    )
    api_key: str


OPENAI_TTS_MODELS = ["gpt-4o-mini-tts"]


@register_tts
class OpenAITTSService(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.OPENAI] = ServiceProviders.OPENAI
    model: str = Field(
        default="gpt-4o-mini-tts", json_schema_extra={"examples": OPENAI_TTS_MODELS}
    )
    voice: str = "alloy"
    api_key: str


DOGRAH_TTS_MODELS = ["default"]


@register_tts
class DograhTTSService(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.DOGRAH] = ServiceProviders.DOGRAH
    model: str = Field(
        default="default", json_schema_extra={"examples": DOGRAH_TTS_MODELS}
    )
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed of the voice")
    api_key: str


CARTESIA_TTS_MODELS = [
    "sonic-3",
    "sonic-3-2026-01-12",
    "sonic-3-latest",
    "sonic-2.1",
    "sonic-english",
    "sonic-multilingual"
]

CARTESIA_EMOTIONS = [
    "neutral",
    "calm",
    "angry",
    "excited",
    "content",
    "sad",
    "scared",
]

CARTESIA_LANGUAGES = [
    "en", "fr", "de", "es", "pt", "zh", "ja", "hi", "it", "ko", "nl", "pl", "ru",
    "sv", "tr", "tl", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms", "sk", "da",
    "ta", "uk", "hu", "no", "vi", "bn", "th", "he", "ka", "id", "te", "gu", "kn",
    "ml", "mr", "pa"
]

@register_tts
class CartesiaTTSConfiguration(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.CARTESIA] = ServiceProviders.CARTESIA
    model: str = Field(
        default="sonic-3", json_schema_extra={"examples": CARTESIA_TTS_MODELS}
    )
    voice: str = Field(default="a167e0f3-df7e-4d52-a9c3-f949145571bd")
    language: str = Field(
        default="en", json_schema_extra={"examples": CARTESIA_LANGUAGES}
    )
    emotion: str = Field(
        default="neutral", json_schema_extra={"examples": CARTESIA_EMOTIONS}
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    telephony_sample_rate: int = Field(
        default=16000,
        json_schema_extra={"examples": [8000, 16000, 22050, 44100]},
        description="Sample rate for telephony calls (8000, 16000, 22050, or 44100 Hz)",
    )
    web_sample_rate: int = Field(
        default=44100,
        json_schema_extra={"examples": [16000, 22050, 44100]},
        description="Sample rate for web/WebRTC calls (up to 44100 Hz)",
    )
    api_key: str


SARVAM_TTS_MODELS = ["bulbul:v2", "bulbul:v3"]
SARVAM_V2_VOICES = [
    "anushka",
    "manisha",
    "vidya",
    "arya",
    "abhilash",
    "karun",
    "hitesh",
]
SARVAM_V3_VOICES = [
    "shubh",
    "aditya",
    "ritu",
    "priya",
    "neha",
    "rahul",
    "pooja",
    "rohan",
    "simran",
    "kavya",
    "amit",
    "dev",
    "ishita",
    "shreya",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "roopa",
    "kabir",
    "aayan",
    "ashutosh",
    "advait",
    "amelia",
    "sophia",
    "anand",
    "tanya",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "shruti",
    "suhani",
    "mohit",
    "kavitha",
    "rehan",
    "soham",
    "rupali",
]
SARVAM_LANGUAGES = [
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "as-IN",
]


@register_tts
class SarvamTTSConfiguration(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.SARVAM] = ServiceProviders.SARVAM
    model: str = Field(
        default="bulbul:v2", json_schema_extra={"examples": SARVAM_TTS_MODELS}
    )
    voice: str = Field(
        default="anushka",
        json_schema_extra={
            "examples": SARVAM_V2_VOICES,
            "model_options": {
                "bulbul:v2": SARVAM_V2_VOICES,
                "bulbul:v3": SARVAM_V3_VOICES,
            },
        },
    )
    language: str = Field(
        default="hi-IN", json_schema_extra={"examples": SARVAM_LANGUAGES}
    )
    api_key: str

SMALLEST_AI_TTS_MODELS = ["lightning-v3.1", "lightning-v2"]
SMALLEST_AI_LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
]
SMALLEST_AI_TELEPHONY_SAMPLE_RATES = [8000, 16000]
SMALLEST_AI_WEB_SAMPLE_RATES = [8000, 16000, 22050, 44100]


@register_tts
class SmallestAITTSConfiguration(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.SMALLEST_AI] = ServiceProviders.SMALLEST_AI
    model: str = Field(
        default="lightning-v3.1", json_schema_extra={"examples": SMALLEST_AI_TTS_MODELS}
    )
    voice: str = Field(
        default="ryan", description="Voice ID to use for TTS"
    )
    language: str = Field(
        default="hi", json_schema_extra={"examples": SMALLEST_AI_LANGUAGES}
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    max_buffer_flush_ms: int = Field(default=0, ge=0)
    consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    enhancement: int = Field(default=1, ge=0, le=1)
    similarity: float = Field(default=0, ge=0.0, le=1.0)
    telephony_sample_rate: int = Field(
        default=8000,
        json_schema_extra={"examples": SMALLEST_AI_TELEPHONY_SAMPLE_RATES},
        description="Sample rate for telephony calls (8000 or 16000 Hz)"
    )
    web_sample_rate: int = Field(
        default=16000,
        json_schema_extra={"examples": SMALLEST_AI_WEB_SAMPLE_RATES},
        description="Sample rate for web/WebRTC calls (up to 44100 Hz)"
    )
    api_key: str


VOICEMAKER_TTS_VOICES = [
    "ai3-Jony",
    "ai3-Aria",
    "ai3-Jenny",
    "ai3-Sonia",
    "ai3-Guy",
]

VOICEMAKER_TTS_LANGUAGES = [
    "en-US", "en-GB", "en-AU", "en-CA", "en-IN",
    "hi-IN", "multi-lang",
    "fr-FR", "fr-CA",
    "de-DE", "es-ES", "es-US",
    "pt-BR", "pt-PT",
    "it-IT", "ja-JP", "ko-KR",
    "zh-CN", "zh-TW",
    "ar-SA", "nl-NL", "pl-PL",
    "ru-RU", "sv-SE", "tr-TR",
]

VOICEMAKER_OUTPUT_FORMATS = ["mp3", "wav", "ogg", "opus", "aac", "ulaw", "alaw"]
VOICEMAKER_TELEPHONY_SAMPLE_RATES = [8000, 16000, 22050, 24000]
VOICEMAKER_WEB_SAMPLE_RATES = [22050, 24000, 44100, 48000]
VOICEMAKER_PRO_ENGINES = ["highres", "turbo", "expressive"]
VOICEMAKER_ACCENT_CODES = [
    "en-US", "en-GB", "en-AU", "en-CA", "en-IN",
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "gu-IN", "mr-IN", "pa-IN",
    "fr-FR", "fr-CA", "de-DE", "es-ES", "es-US", "pt-BR", "it-IT", "ja-JP"
]

MURF_TTS_MODELS = ["FALCON"]
MURF_REGIONS = [
    "global",
    "us-east",
    "us-west",
    "in",
    "ca",
    "kr",
    "me",
    "jp",
    "au",
    "eu-central",
    "uk",
    "sa-east",
]
MURF_TELEPHONY_SAMPLE_RATES = [8000, 16000]
MURF_WEB_SAMPLE_RATES = [16000, 24000]


@register_tts
class VoicemakerTTSConfiguration(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.VOICEMAKER] = ServiceProviders.VOICEMAKER
    model: str = Field(
        default="neural",
        json_schema_extra={"examples": ["neural", "standard"]},
        description="Voicemaker engine type (neural or standard)",
    )
    voice: str = Field(
        default="ai3-Jony",
        description="Voicemaker VoiceId (e.g. ai3-Jony)",
    )
    language: str = Field(
        default="en-US",
        json_schema_extra={"examples": VOICEMAKER_TTS_LANGUAGES},
        description="Language code (e.g. en-US, hi-IN, multi-lang)",
    )
    telephony_sample_rate: int = Field(
        default=8000,
        json_schema_extra={"examples": VOICEMAKER_TELEPHONY_SAMPLE_RATES},
        description="Sample rate for telephony calls (8000, 16000, 22050, 24000 Hz)",
    )
    web_sample_rate: int = Field(
        default=48000,
        json_schema_extra={"examples": VOICEMAKER_WEB_SAMPLE_RATES},
        description="Sample rate for web/WebRTC calls (22050, 24000, 44100, 48000 Hz)",
    )
    # Optional quality tuning – stored as numbers, converted to strings when calling the API
    master_speed: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description="Speed adjustment: -100 to 100 (0 = normal)",
    )
    master_pitch: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description="Pitch adjustment: -100 to 100 (0 = normal)",
    )
    master_volume: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0,
        description="Volume adjustment: -20 to 20 (0 = normal)",
    )
    # ProPlus-only (optional floats so the UI can pass numbers directly)
    stability: Union[float, None] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Stability 0-100 (ProPlus voices only)",
    )
    similarity: Union[float, None] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Similarity 0-100 (ProPlus voices only)",
    )
    pro_engine: Union[str, None] = Field(
        default=None,
        json_schema_extra={"examples": VOICEMAKER_PRO_ENGINES},
        description="Pro engine: turbo, highres, or expressive (ProPlus voices only)",
    )
    accent_code: Union[str, None] = Field(
        default=None,
        json_schema_extra={"examples": VOICEMAKER_ACCENT_CODES},
        description="Accent code for multilingual voices (e.g. en-US, fr-FR)",
    )
    api_key: str


@register_tts
class MurfTTSConfiguration(BaseTTSConfiguration):
    provider: Literal[ServiceProviders.MURF] = ServiceProviders.MURF
    model: str = Field(
        default="FALCON",
        json_schema_extra={"examples": MURF_TTS_MODELS},
        description="Murf realtime TTS model",
    )
    voice: str = Field(
        default="Matthew",
        description="Murf voice ID",
    )
    locale: str = Field(
        default="en-US",
        description="Locale supported by the selected Murf voice",
    )
    style: str = Field(
        default="Conversation",
        description="Speech style supported by the selected Murf voice",
    )
    rate: int = Field(
        default=0,
        ge=-50,
        le=50,
        description="Speech rate from -50 to 50",
    )
    pitch: int = Field(
        default=0,
        ge=-50,
        le=50,
        description="Speech pitch from -50 to 50",
    )
    min_buffer_size: int = Field(
        default=40,
        ge=40,
        le=160,
        description="Minimum characters Murf buffers before synthesizing incomplete text",
    )
    max_buffer_delay_in_ms: int = Field(
        default=300,
        ge=0,
        le=1000,
        description="Maximum wait before Murf synthesizes incomplete text",
    )
    region: str = Field(
        default="us-east",
        json_schema_extra={"examples": MURF_REGIONS},
        description="Murf streaming region",
    )
    telephony_sample_rate: int = Field(
        default=8000,
        json_schema_extra={"examples": MURF_TELEPHONY_SAMPLE_RATES},
        description="Sample rate for telephony calls",
    )
    web_sample_rate: int = Field(
        default=16000,
        json_schema_extra={"examples": MURF_WEB_SAMPLE_RATES},
        description="Sample rate for web/WebRTC calls",
    )
    api_key: str


TTSConfig = Annotated[
    Union[
        DeepgramTTSConfiguration,
        FishTTSConfiguration,
        OpenAITTSService,
        ElevenlabsTTSConfiguration,
        CartesiaTTSConfiguration,
        DograhTTSService,
        SarvamTTSConfiguration,
        SmallestAITTSConfiguration,
        VoicemakerTTSConfiguration,
        MurfTTSConfiguration,
    ],
    Field(discriminator="provider"),
]

###################################################### STT ########################################################################


DEEPGRAM_STT_MODELS = ["nova-3-general", "flux-general-en"]
DEEPGRAM_LANGUAGES = [
    "multi",
    "ar",
    "ar-AE",
    "ar-SA",
    "ar-QA",
    "ar-KW",
    "ar-SY",
    "ar-LB",
    "ar-PS",
    "ar-JO",
    "ar-EG",
    "ar-SD",
    "ar-TD",
    "ar-MA",
    "ar-DZ",
    "ar-TN",
    "ar-IQ",
    "ar-IR",
    "be",
    "bn",
    "bs",
    "bg",
    "ca",
    "cs",
    "da",
    "da-DK",
    "de",
    "de-CH",
    "el",
    "en",
    "en-US",
    "en-AU",
    "en-GB",
    "en-IN",
    "en-NZ",
    "es",
    "es-419",
    "et",
    "fa",
    "fi",
    "fr",
    "fr-CA",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "kn",
    "ko",
    "ko-KR",
    "lt",
    "lv",
    "mk",
    "mr",
    "ms",
    "nl",
    "nl-BE",
    "no",
    "pl",
    "pt",
    "pt-BR",
    "pt-PT",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "sv-SE",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh-CN",
    "zh-TW",
]


@register_stt
class DeepgramSTTConfiguration(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.DEEPGRAM] = ServiceProviders.DEEPGRAM
    model: str = Field(
        default="nova-3-general", json_schema_extra={"examples": DEEPGRAM_STT_MODELS}
    )
    language: str = Field(
        default="multi",
        json_schema_extra={
            "examples": DEEPGRAM_LANGUAGES,
            "model_options": {
                "nova-3-general": DEEPGRAM_LANGUAGES,
                "flux-general-en": ["en"],
            },
        },
    )
    api_key: str


CARTESIA_STT_MODELS = ["ink-whisper"]
CARTESIA_STT_LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
]


@register_stt
class CartesiaSTTConfiguration(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.CARTESIA] = ServiceProviders.CARTESIA
    model: str = Field(
        default="ink-whisper", json_schema_extra={"examples": CARTESIA_STT_MODELS}
    )
    language: str = Field(
        default="hi", json_schema_extra={"examples": CARTESIA_STT_LANGUAGES}
    )
    api_key: str


OPENAI_STT_MODELS = ["gpt-4o-transcribe"]


@register_stt
class OpenAISTTConfiguration(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.OPENAI] = ServiceProviders.OPENAI
    model: str = Field(
        default="gpt-4o-transcribe", json_schema_extra={"examples": OPENAI_STT_MODELS}
    )
    api_key: str


# Dograh STT Service
DOGRAH_STT_MODELS = ["default"]
DOGRAH_STT_LANGUAGES = DEEPGRAM_LANGUAGES


@register_stt
class DograhSTTService(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.DOGRAH] = ServiceProviders.DOGRAH
    model: str = Field(
        default="default", json_schema_extra={"examples": DOGRAH_STT_MODELS}
    )
    language: str = Field(
        default="multi", json_schema_extra={"examples": DOGRAH_STT_LANGUAGES}
    )
    api_key: str


# Sarvam STT Service
SARVAM_STT_MODELS = ["saaras:v3", "saarika:v2.5", "saaras:v2"]
SARVAM_MODE_OPTIONS = ["transcribe", "translate", "verbatim", "translit", "codemix"]


@register_stt
class SarvamSTTConfiguration(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.SARVAM] = ServiceProviders.SARVAM
    model: str = Field(
        default="saaras:v3", json_schema_extra={"examples": SARVAM_STT_MODELS}
    )
    language: str = Field(
        default="unknown", json_schema_extra={"examples": SARVAM_LANGUAGES + ["unknown"]}
    )
    mode: str = Field(
        default="transcribe", json_schema_extra={"examples": SARVAM_MODE_OPTIONS}
    )
    api_key: str


# Speechmatics STT Service
SPEECHMATICS_STT_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "ja",
    "ko",
    "zh",
    "ru",
    "ar",
    "hi",
    "pl",
    "tr",
    "vi",
    "th",
    "id",
    "ms",
    "sv",
    "da",
    "no",
    "fi",
]


@register_stt
class SpeechmaticsSTTConfiguration(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.SPEECHMATICS] = ServiceProviders.SPEECHMATICS
    model: str = Field(
        default="enhanced", description="Operating point: standard or enhanced"
    )
    language: str = Field(
        default="en", json_schema_extra={"examples": SPEECHMATICS_STT_LANGUAGES}
    )
    api_key: str


SONIOX_STT_MODELS = ["stt-rt-v4", "stt-rt-v3"]
SONIOX_LANGUAGE_HINTS = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lo",
    "lt",
    "lv",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
]

@register_stt
class SonioxSTTConfiguration(BaseSTTConfiguration):
    provider: Literal[ServiceProviders.SONIOX] = ServiceProviders.SONIOX
    model: str = Field(
        default="stt-rt-v4", json_schema_extra={"examples": SONIOX_STT_MODELS}
    )
    language_hints: list[str] = Field(
        default_factory=list,
        json_schema_extra={"examples": SONIOX_LANGUAGE_HINTS},
        description="Expected ISO language codes (e.g. en, es).",
    )
    language_hints_strict: bool = Field(
        default=False,
        description=(
            "Restrict recognition to language_hints. Best-effort; single-language "
            "selection is most robust."
        ),
    )
    api_key: str


STTConfig = Annotated[
    Union[
        DeepgramSTTConfiguration,
        CartesiaSTTConfiguration,
        OpenAISTTConfiguration,
        DograhSTTService,
        SpeechmaticsSTTConfiguration,
        SarvamSTTConfiguration,
        SonioxSTTConfiguration,
    ],
    Field(discriminator="provider"),
]

###################################################### EMBEDDINGS ########################################################################

OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small"]


@register_embeddings
class OpenAIEmbeddingsConfiguration(BaseEmbeddingsConfiguration):
    provider: Literal[ServiceProviders.OPENAI] = ServiceProviders.OPENAI
    model: str = Field(
        default="text-embedding-3-small",
        json_schema_extra={"examples": OPENAI_EMBEDDING_MODELS},
    )
    api_key: str


OPENROUTER_EMBEDDING_MODELS = ["openai/text-embedding-3-small"]


@register_embeddings
class OpenRouterEmbeddingsConfiguration(BaseEmbeddingsConfiguration):
    provider: Literal[ServiceProviders.OPENROUTER] = ServiceProviders.OPENROUTER
    model: str = Field(
        default="openai/text-embedding-3-small",
        json_schema_extra={"examples": OPENROUTER_EMBEDDING_MODELS},
    )
    api_key: str
    base_url: str = Field(default="https://openrouter.ai/api/v1")


EmbeddingsConfig = Annotated[
    Union[OpenAIEmbeddingsConfiguration, OpenRouterEmbeddingsConfiguration],
    Field(discriminator="provider"),
]

ServiceConfig = Annotated[
    Union[LLMConfig, TTSConfig, STTConfig, EmbeddingsConfig],
    Field(discriminator="provider"),
]
