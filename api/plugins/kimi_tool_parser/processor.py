import json
import re

from loguru import logger

from pipecat.frames.frames import (
    TextFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
# We use pipecat's new LLMToolCallFrame style if available, else we handle it correctly.
# Currently pipecat has varying function call frame formats but Kimi outputs it exactly like Pipecat's internal format
# if we just parse the json.
try:
    from pipecat.frames.frames import LLMFullResponseStartFrame, LLMFullResponseEndFrame, UserStartedSpeakingFrame, InterruptionFrame
except ImportError:
    pass

def extract_tool_call_info(tool_call_rsp: str):
    tool_calls = []
    
    # 1. Moonshot Native Format
    if '<|tool_calls_section_begin|>' in tool_call_rsp:
        pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
        sections = re.findall(pattern, tool_call_rsp, re.DOTALL)
        if sections:
            func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
            for match in re.findall(func_call_pattern, sections[0], re.DOTALL):
                function_id, function_args = match
                function_name = function_id.split('.')[1].split(':')[0]
                tool_calls.append({
                    "id": function_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_args
                    }
                })
                
    # 2. DeepInfra / vLLM fallback format
    if '<function_calls>' in tool_call_rsp:
        pattern = r"<function_calls>(.*?)</function_calls>"
        sections = re.findall(pattern, tool_call_rsp, re.DOTALL)
        if sections:
            import uuid
            block = sections[0]
            invokes = re.findall(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', block, re.DOTALL)
            for func_name, args_block in invokes:
                args = {}
                if args_block and args_block.strip():
                    arg_matches = re.findall(r'<([^>]+)>(.*?)</\1>', args_block, re.DOTALL)
                    for k, v in arg_matches:
                        args[k] = v.strip()
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })

    return tool_calls

class KimiToolCallInterceptor(FrameProcessor):
    """
    Kimi K2 outputs raw string tags instead of OpenAI standard JSON tool calls.
    This processor captures TextFrames that start Kimi tags, buffers them,
    parses them upon completion, and emits standard pipecat LLM Tool Call frames.
    """
    def __init__(self):
        super().__init__()
        self._is_buffering_tool = False
        self._tool_call_buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            text = frame.text

            # If we see the start tag in this chunk or are already buffering
            is_moonshot_tag = "<|tool_calls_section_begin|>" in text
            is_vllm_tag = "<function_calls>" in text

            if is_moonshot_tag or is_vllm_tag or self._is_buffering_tool:
                if not self._is_buffering_tool:
                    self._is_buffering_tool = True
                    self._tool_call_buffer = ""
                    logger.debug(f"KimiToolCallInterceptor: Triggered, buffering tool call stream.")
                
                self._tool_call_buffer += text
                
                # If we've hit the end tag, parse and emit
                has_moonshot_end = "<|tool_calls_section_end|>" in self._tool_call_buffer
                has_vllm_end = "</function_calls>" in self._tool_call_buffer

                if has_moonshot_end or has_vllm_end:
                    logger.debug(f"KimiToolCallInterceptor: Buffer complete, parsing...")
                    tool_calls = extract_tool_call_info(self._tool_call_buffer)
                    if tool_calls:
                        # Attempt to map to Pipecat frames
                        try:
                            # Try the single tool call frame (newer Pipecat)
                            from pipecat.frames.frames import LLMToolCallFrame
                            for tc in tool_calls:
                                tool_name = tc.get("function", {}).get("name", "")
                                tool_args = tc.get("function", {}).get("arguments", "{}")
                                tool_id = tc.get("id", "")
                                logger.info(f"KimiToolCallInterceptor: Emitting LLMToolCallFrame({tool_name}, args={tool_args})")
                                await self.push_frame(LLMToolCallFrame(
                                    function_name=tool_name,
                                    tool_call_id=tool_id,
                                    arguments=tool_args
                                ), direction)
                        except ImportError:
                            # Try the older FunctionCall format
                            try:
                                from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallFrame
                                for tc in tool_calls:
                                    tool_name = tc.get("function", {}).get("name", "")
                                    args_raw = tc.get("function", {}).get("arguments", "{}")
                                    tool_args = json.loads(args_raw) if args_raw else {}
                                    tool_id = tc.get("id", "")
                                    logger.info(f"KimiToolCallInterceptor: Emitting FunctionCallFrame({tool_name})")
                                    await self.push_frame(FunctionCallInProgressFrame(
                                        function_name=tool_name,
                                        tool_call_id=tool_id,
                                        arguments=args_raw
                                    ), direction)
                                    await self.push_frame(FunctionCallFrame(
                                        function_name=tool_name,
                                        tool_call_id=tool_id,
                                        arguments=tool_args
                                    ), direction)
                            except Exception as e:
                                logger.error(f"KimiToolCallInterceptor: Failed to construct Pipecat tool frames: {e}")
                    else:
                        logger.warning(f"KimiToolCallInterceptor: Failed to parse buffered tool call tags: {self._tool_call_buffer}")

                    # Reset state
                    self._is_buffering_tool = False
                    self._tool_call_buffer = ""
                
                # Consume this TextFrame so it doesn't leak into the TTS!
                return
            
            # Normal conversation text, pass down to TTS
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
