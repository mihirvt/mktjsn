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
                        
            params_copy = dict(params_from_context)
            params_copy["messages"] = messages
            params.update(params_copy)
        else:
            params.update(params_from_context)

        return params

    async def get_chat_completions(self, params_from_context):
        chunks = await super().get_chat_completions(params_from_context)
        
        # Intercept and wrap the stream to intercept leaked XML tool calls for Kimi models
        if getattr(self, "model_name", "").lower().startswith("moonshotai/kimi"):
            async def _xml_to_tool_call_stream(stream):
                xml_buffer = ""
                
                import json
                import re
                import copy
                try:
                    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
                except ImportError:
                    ChoiceDeltaToolCall = None

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        xml_buffer += content
                        
                        while True:
                            start_idx = xml_buffer.find("<function_calls>")
                            if start_idx == -1:
                                start_idx = xml_buffer.find("<function_call>")
                                
                            if start_idx != -1:
                                end_idx1 = xml_buffer.find("</function_calls>", start_idx)
                                end_idx2 = xml_buffer.find("</function_call>", start_idx)
                                end_idx = max(end_idx1, end_idx2)
                                
                                if end_idx != -1:
                                    close_idx = xml_buffer.find(">", end_idx)
                                    if close_idx != -1:
                                        full_xml = xml_buffer[start_idx:close_idx+1]
                                        
                                        match = re.search(r'<invoke\s+name=["\']([^"\']+)["\']', full_xml)
                                        if match and ChoiceDeltaToolCall is not None:
                                            func_name = match.group(1)
                                            args_dict = {}
                                            param_matches = re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']>([^<]*)</parameter>', full_xml)
                                            for pmatch in param_matches:
                                                args_dict[pmatch.group(1)] = pmatch.group(2)
                                                
                                            import uuid
                                            tc_func = ChoiceDeltaToolCallFunction(name=func_name, arguments=json.dumps(args_dict))
                                            tc = ChoiceDeltaToolCall(index=0, id=f"call_{uuid.uuid4().hex[:8]}", type="function", function=tc_func)
                                            
                                            tc_chunk = copy.deepcopy(chunk)
                                            tc_chunk.choices[0].delta.content = None
                                            if hasattr(tc_chunk.choices[0].delta, "tool_calls"):
                                                tc_chunk.choices[0].delta.tool_calls = [tc]
                                            else:
                                                setattr(tc_chunk.choices[0].delta, "tool_calls", [tc])
                                            yield tc_chunk
                                            
                                        xml_buffer = xml_buffer[:start_idx] + xml_buffer[close_idx+1:]
                                        continue
                            break
                            
                        safe_text = ""
                        if "<function_" in xml_buffer:
                            idx = xml_buffer.find("<function_")
                            safe_text = xml_buffer[:idx]
                            xml_buffer = xml_buffer[idx:]
                        else:
                            last_open = xml_buffer.rfind("<")
                            if last_open != -1 and ("<function_calls>"[:len(xml_buffer)-last_open] == xml_buffer[last_open:] or "<function_call>"[:len(xml_buffer)-last_open] == xml_buffer[last_open:]):
                                safe_text = xml_buffer[:last_open]
                                xml_buffer = xml_buffer[last_open:]
                            else:
                                safe_text = xml_buffer
                                xml_buffer = ""
                                
                        if safe_text:
                            text_chunk = copy.deepcopy(chunk)
                            text_chunk.choices[0].delta.content = safe_text
                            if hasattr(text_chunk.choices[0].delta, "tool_calls"):
                                text_chunk.choices[0].delta.tool_calls = None
                            yield text_chunk
                            
                    else:
                        yield chunk
            return _xml_to_tool_call_stream(chunks)
        
        return chunks
