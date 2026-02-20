#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger
from typing import Optional, Dict, Any

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.llm import OpenAILLMService


class SarvamLLMService(OpenAILLMService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai/v1",
        model: str = "sarvam-m",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict:
        """Build parameters for Sarvam chat completion request.

        Includes support for reasoning_effort and Sarvam-specific parameters.
        Carefully filters out unsupported parameters.
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
        
        return params
