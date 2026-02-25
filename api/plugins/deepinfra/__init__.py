#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import hashlib

from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.llm import OpenAILLMService


class DeepInfraLLMService(OpenAILLMService):
    """DeepInfra LLM service using their OpenAI-compatible API.

    DeepInfra API: https://api.deepinfra.com/v1/openai/chat/completions
    Supports streaming, function calling, reasoning_effort, and prompt caching.

    Prompt Caching:
        DeepInfra supports server-side prefix caching via `prompt_cache_key`.
        We hash the system prompt content to generate a stable cache key so that
        identical prompts across different calls reuse cached KV computations,
        reducing token costs. When a user edits the prompt in the dashboard,
        the hash changes automatically → new cache entry → no stale data.
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

    @staticmethod
    def _generate_prompt_cache_key(messages: list) -> str:
        """Generate a stable cache key from the system prompt.

        Uses SHA-256 hash of the system message content so that:
        - Same prompt across calls → same key → DeepInfra reuses cached KV computation
        - Prompt edited in dashboard → different hash → new cache entry
        """
        # Extract system message content for cache key
        system_content = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_content += content
                elif isinstance(content, list):
                    # Handle content parts format
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_content += part.get("text", "")
                break  # Only use the first system message

        if not system_content:
            return ""

        return hashlib.sha256(system_content.encode("utf-8")).hexdigest()[:16]

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict:
        """Build parameters for DeepInfra chat completion request.

        Includes support for reasoning_effort, prompt caching, and
        DeepInfra-specific parameters. Carefully filters out unsupported params.
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

        # Handle extra (reasoning_effort)
        extra = self._settings.get("extra", {})
        if "reasoning_effort" in extra:
            params["reasoning_effort"] = extra["reasoning_effort"]

        # Add max_tokens if set (standard OpenAI param)
        if self._settings["max_tokens"]:
            params["max_tokens"] = self._settings["max_tokens"]
        if self._settings["max_completion_tokens"]:
            params["max_completion_tokens"] = self._settings["max_completion_tokens"]

        # Add messages, tools, tool_choice from context (OpenAILLMInvocationParams is a dict)
        params.update(params_from_context)

        # Generate prompt_cache_key from system prompt for DeepInfra server-side
        # prefix caching. This saves token costs when the same agent prompt is
        # used across multiple calls (which is the common case).
        messages = params.get("messages", [])
        if messages:
            cache_key = self._generate_prompt_cache_key(messages)
            if cache_key:
                params["prompt_cache_key"] = cache_key

        return params
