from typing import Dict, Optional, TypedDict

import openai
from deepgram import (
    DeepgramClient,
    LiveOptions,
)
from groq import Groq
from loguru import logger

# try:
#     from pyneuphonic import Neuphonic
# except ImportError:
#     Neuphonic = None
from api.schemas.user_configuration import (
    UserConfiguration,
)
from api.services.configuration.registry import ServiceConfig, ServiceProviders


class APIKeyStatus(TypedDict):
    model: str
    message: str


class APIKeyStatusResponse(TypedDict):
    status: list[APIKeyStatus]


class UserConfigurationValidator:
    def __init__(self):
        self._provider_api_key_validity_status: Dict[str, bool] = {}
        self._validator_map = {
            ServiceProviders.OPENAI.value: self._check_openai_api_key,
            ServiceProviders.DEEPGRAM.value: self._check_deepgram_api_key,
            ServiceProviders.GROQ.value: self._check_groq_api_key,
            ServiceProviders.OPENROUTER.value: self._check_openrouter_api_key,
            ServiceProviders.ELEVENLABS.value: self._validate_elevenlabs_api_key,
            ServiceProviders.GOOGLE.value: self._check_google_api_key,
            ServiceProviders.AZURE.value: self._check_azure_api_key,
            ServiceProviders.CARTESIA.value: self._check_cartesia_api_key,
            ServiceProviders.DOGRAH.value: self._check_dograh_api_key,
            ServiceProviders.SARVAM.value: self._check_sarvam_api_key,
            ServiceProviders.SPEECHMATICS.value: self._check_speechmatics_api_key,
            ServiceProviders.SMALLEST_AI.value: self._check_smallest_ai_api_key,
            ServiceProviders.DEEPINFRA.value: self._check_deepinfra_api_key,
            ServiceProviders.SONIOX.value: self._check_soniox_api_key,
        }

    async def validate(self, configuration: UserConfiguration) -> APIKeyStatusResponse:
        status_list = []

        status_list.extend(self._validate_service(configuration.llm, "llm"))
        status_list.extend(self._validate_service(configuration.stt, "stt"))
        status_list.extend(self._validate_service(configuration.tts, "tts"))
        # Embeddings is optional - only validate if configured
        status_list.extend(
            self._validate_service(
                configuration.embeddings, "embeddings", required=False
            )
        )

        if status_list:
            raise ValueError(status_list)

        return {"status": [{"model": "all", "message": "ok"}]}

    def _validate_service(
        self,
        service_config: Optional[ServiceConfig],
        service_name: str,
        required: bool = True,
    ) -> list[APIKeyStatus]:
        """Validate a service configuration and return any error statuses."""
        if not service_config:
            if required:
                return [{"model": service_name, "message": "API key is missing"}]
            return []  # Optional service not configured is OK

        provider = service_config.provider
        api_key = service_config.api_key
        
        provider_val = provider.value if hasattr(provider, "value") else provider

        if not self._check_api_key(provider, api_key):
            return [{"model": service_name, "message": f"Invalid {provider_val} API key"}]

        return []

    def _check_api_key(self, provider: str, api_key: str) -> bool:
        """Check if an API key for a provider is valid."""
        # Provider might be an Enum object at runtime instead of a string
        provider_val = provider.value if hasattr(provider, "value") else provider
        normalized_api_key = api_key.strip() if isinstance(api_key, str) else ""
        if not normalized_api_key:
            return False

        validator = self._validator_map.get(provider_val)
        if not validator:
            logger.warning(f"No API key validator configured for provider '{provider_val}'")
            return False

        return validator(provider_val, normalized_api_key)

    def _check_openai_api_key(self, model: str, api_key: str) -> bool:
        cache_key = f"{model}:{api_key}"
        if cache_key in self._provider_api_key_validity_status:
            return self._provider_api_key_validity_status[cache_key]

        client = openai.OpenAI(api_key=api_key)
        try:
            client.models.list()
            self._provider_api_key_validity_status[cache_key] = True
        except openai.AuthenticationError:
            self._provider_api_key_validity_status[cache_key] = False
        return self._provider_api_key_validity_status[cache_key]

    def _check_deepgram_api_key(self, model: str, api_key: str) -> bool:
        cache_key = f"{model}:{api_key}"
        if cache_key in self._provider_api_key_validity_status:
            return self._provider_api_key_validity_status[cache_key]

        deepgram = DeepgramClient(api_key)
        dg_connection = deepgram.listen.websocket.v("1")

        try:
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
            )

            connected = dg_connection.start(options)
            self._provider_api_key_validity_status[cache_key] = connected
        finally:
            dg_connection.finish()
        return self._provider_api_key_validity_status[cache_key]

    def _check_groq_api_key(self, model: str, api_key: str) -> bool:
        cache_key = f"{model}:{api_key}"
        if cache_key in self._provider_api_key_validity_status:
            return self._provider_api_key_validity_status[cache_key]

        try:
            client = Groq(api_key=api_key)
            client.models.list()
            self._provider_api_key_validity_status[cache_key] = True
        except Exception as e:
            # Avoid false "invalid key" errors for transient network/provider issues.
            # Treat explicit auth failures as invalid; soft-pass other validation failures.
            message = str(e).lower()
            auth_failure_markers = [
                "invalid api key",
                "authentication",
                "unauthorized",
                "forbidden",
                "permission denied",
                "401",
                "403",
            ]
            is_auth_failure = any(marker in message for marker in auth_failure_markers)
            self._provider_api_key_validity_status[cache_key] = not is_auth_failure
            logger.warning(
                "Groq key validation failed with non-auth-safe fallback. "
                f"auth_failure={is_auth_failure}, error={e}"
            )
        return self._provider_api_key_validity_status[cache_key]

    def _validate_elevenlabs_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_google_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_azure_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_cartesia_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_dograh_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_sarvam_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_openrouter_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_speechmatics_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_smallest_ai_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_deepinfra_api_key(self, model: str, api_key: str) -> bool:
        return True

    def _check_soniox_api_key(self, model: str, api_key: str) -> bool:
        return True
