#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.llm import OpenAILLMService


class DeepInfraLLMService(OpenAILLMService):
    """DeepInfra LLM service using their OpenAI-compatible API.

    DeepInfra API: https://api.deepinfra.com/v1/openai/chat/completions
    Supports streaming, function calling, and reasoning_effort parameter.

    Caching:
        DeepInfra uses automatic prefix caching via vLLM internally —
        no API parameter needed. Identical prompt prefixes across requests
        are automatically cached server-side, giving a ~50% discount
        on cached tokens. This works transparently without any client-side
        configuration.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        model: str = "moonshotai/Kimi-K2.5",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict:
        """Build parameters for DeepInfra chat completion request.

        Only includes parameters that DeepInfra's OpenAI-compatible API
        actually supports. Carefully filters out unsupported parameters.
        """
        # Start with essential params
        params = {
            "model": self.model_name,
            "stream": True,
        }

        # Add temperature if set
        if self._settings["temperature"] is not None:
            params["temperature"] = self._settings["temperature"]

        # Add top_p if set
        if self._settings["top_p"] is not None:
            params["top_p"] = self._settings["top_p"]

        # Always send reasoning_effort (DeepInfra supports this for
        # compatible models and ignores it for others). Sending "none"
        # explicitly ensures no surprise reasoning delays.
        extra = self._settings.get("extra", {})
        reasoning_effort = extra.get("reasoning_effort", "none")
        params["reasoning_effort"] = reasoning_effort

        # Add max_tokens if set (standard OpenAI param)
        if self._settings["max_tokens"]:
            params["max_tokens"] = self._settings["max_tokens"]
        if self._settings["max_completion_tokens"]:
            params["max_completion_tokens"] = self._settings["max_completion_tokens"]

        # Add messages, tools, tool_choice from context
        params.update(params_from_context)

        return params
