#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq LLM Service implementation using OpenAI-compatible interface."""

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class GroqLLMService(OpenAILLMService):
    """A service for interacting with Groq's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Groq's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        **kwargs,
    ):
        """Initialize Groq LLM service.

        Args:
            api_key: The API key for accessing Groq's API.
            base_url: The base URL for Groq API. Defaults to "https://api.groq.com/openai/v1".
            model: The model identifier to use. Defaults to "llama-3.3-70b-versatile".
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Groq API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance api_key.
            base_url: Base URL for the API. If None, uses instance base_url.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Groq's API.
        """
        logger.debug(f"Creating Groq client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(
        self, params_from_context
    ) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        
        # Handle Kimi K2's requirement for sequential tool call IDs
        # (functions.func_name:idx) instead of standard random UUIDs
        if getattr(self, "model_name", "").lower().startswith("moonshotai/kimi"):
            import copy
            messages = copy.deepcopy(params_from_context.get("messages", []))
            
            id_map = {}
            global_idx = 0
            
            for msg in messages:
                # Assistant messages making tool_calls
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        if tc.get("type") == "function":
                            func_name = tc.get("function", {}).get("name", "")
                            old_id = tc.get("id")
                            if old_id and old_id not in id_map:
                                new_id = f"functions.{func_name}:{global_idx}"
                                id_map[old_id] = new_id
                                global_idx += 1
                            if old_id in id_map:
                                tc["id"] = id_map[old_id]
                
                # Tool messages returning the results
                elif msg.get("role") == "tool":
                    old_id = msg.get("tool_call_id")
                    if old_id in id_map:
                        msg["tool_call_id"] = id_map[old_id]
                        
            params["messages"] = messages

        return params
