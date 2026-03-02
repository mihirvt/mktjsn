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
    from pipecat.frames.frames import (
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
        UserStartedSpeakingFrame,
        InterruptionFrame,
        LLMContextFrame
    )
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
    from pipecat.services.llm_service import FunctionCallFromLLM
    from pipecat.frames.frames import FunctionCallsFromLLMInfoFrame
except ImportError:
    pass

def extract_tool_call_info(tool_call_rsp: str):
    tool_calls = []
    
    # 1. Moonshot Native Format
    if '<|tool_call_begin|>' in tool_call_rsp or '<|tool_calls_section_begin|>' in tool_call_rsp:
        # Some providers may omit section end tokens. Parse tool_call blocks directly from
        # the full text instead of requiring wrapper tags to be complete.
        func_call_pattern = (
            r"\<\|tool_call_begin\|\>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*"
            r"\<\|tool_call_argument_begin\|\>\s*(?P<function_arguments>.*?)\s*"
            r"\<\|tool_call_end\|\>"
        )
        for match in re.findall(func_call_pattern, tool_call_rsp, re.DOTALL):
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

            # Also support self-closing invoke tags with no args.
            self_closing_invokes = re.findall(r'<invoke\s+name="([^"]+)"\s*/>', block, re.DOTALL)
            for func_name in self_closing_invokes:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps({})
                    }
                })
        else:
            # Fallback when wrapper is incomplete/missing close tag:
            # parse invoke blocks directly from full text.
            import uuid
            invokes = re.findall(
                r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>',
                tool_call_rsp,
                re.DOTALL,
            )
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

            self_closing_invokes = re.findall(
                r'<invoke\s+name="([^"]+)"\s*/>',
                tool_call_rsp,
                re.DOTALL,
            )
            for func_name in self_closing_invokes:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps({})
                    }
                })

    return tool_calls


# Characters that mark the end of a spoken sentence/clause.
# We flush the lookahead buffer only at these boundaries so that the preamble
# echoed by Kimi K2 before <function_calls> (which is never a complete sentence)
# stays buffered long enough for the tag to be detected and dropped.
_SENTENCE_END_CHARS = frozenset(['।', '.', '?', '!'])

# Safety valve: if the lookahead buffer exceeds this many chars with no sentence
# boundary, force-flush to avoid starving TTS on legitimately long unpunctuated text.
_MAX_LOOKAHEAD_CHARS = 400


def _last_sentence_boundary(text: str) -> int:
    """Return the index just AFTER the last sentence-ending char, or 0 if none."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] in _SENTENCE_END_CHARS:
            return i + 1
    return 0


class KimiToolCallInterceptor(FrameProcessor):
    """
    Kimi K2 outputs raw string tags instead of OpenAI standard JSON tool calls.
    This processor captures TextFrames that start Kimi tags, buffers them,
    parses them upon completion, and emits standard pipecat LLM Tool Call frames.

    Key design: text is NOT flushed to TTS on every token. Instead it accumulates
    in a lookahead buffer and is released only at sentence boundaries (. ? ! ।).
    This ensures that the preamble Kimi K2 emits before <function_calls> — which
    always echoes the user's words and never ends with sentence punctuation — stays
    buffered until the tag is detected, at which point it is silently dropped.
    """
    def __init__(self, llm=None):
        super().__init__()
        self._llm = llm
        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        self._lookahead_buffer = ""
        self._latest_context = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            try:
                if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
                    self._latest_context = frame.context
            except NameError:
                pass

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            text = frame.text

            # ── Phase 1: already inside a tool call block ──────────────────────
            if self._is_buffering_tool:
                self._tool_call_buffer += text
                
                # If we've hit the end tag, parse and emit
                has_moonshot_end = "<|tool_calls_section_end|>" in self._tool_call_buffer
                has_vllm_end = "</function_calls>" in self._tool_call_buffer

                if has_moonshot_end or has_vllm_end:
                    logger.debug(f"KimiToolCallInterceptor: Buffer complete, parsing...")
                    tool_calls = extract_tool_call_info(self._tool_call_buffer)
                    if tool_calls:
                        if self._llm and hasattr(self._llm, 'run_function_calls'):
                            try:
                                calls_to_run = []
                                for tc in tool_calls:
                                    tool_name = tc.get("function", {}).get("name", "")
                                    args_raw = tc.get("function", {}).get("arguments", "{}")
                                    tool_args = json.loads(args_raw) if args_raw else {}
                                    tool_id = tc.get("id", "")
                                    calls_to_run.append(FunctionCallFromLLM(
                                        context=self._latest_context,
                                        tool_call_id=tool_id,
                                        function_name=tool_name,
                                        arguments=tool_args
                                    ))
                                
                                logger.info(f"KimiToolCallInterceptor: Submitting {len(calls_to_run)} function calls to LLM service")
                                # Send info frame down for observers (e.g. realtime feedback)
                                await self.push_frame(FunctionCallsFromLLMInfoFrame(function_calls=calls_to_run), direction)
                                # Execute function calls on the underlying LLM
                                await self._llm.run_function_calls(calls_to_run)
                            except Exception as e:
                                logger.error(f"KimiToolCallInterceptor: Failed to run function calls: {e}")
                        else:
                            logger.error("KimiToolCallInterceptor: LLM reference missing or does not have run_function_calls")
                    else:
                        logger.warning(f"KimiToolCallInterceptor: Failed to parse buffered tool call tags: {self._tool_call_buffer}")

                    # Reset state
                    self._is_buffering_tool = False
                    self._tool_call_buffer = ""
                
                # Consume this TextFrame — never let tool call text reach TTS
                return

            # ── Phase 2: not yet in a tool call — accumulate lookahead buffer ──
            #
            # We do NOT flush on every token. Instead we accumulate and only flush
            # at sentence-ending characters. This is the core fix: Kimi K2's
            # pre-tool-call preamble never ends with punctuation, so it stays
            # buffered until we detect <function_calls> and can drop it cleanly.
            self._lookahead_buffer += text

            # Check if a complete tool call start tag is now in the buffer
            if "<|tool_calls_section_begin|>" in self._lookahead_buffer or "<function_calls>" in self._lookahead_buffer:
                tag_idx1 = self._lookahead_buffer.find("<|tool_calls_section_begin|>")
                tag_idx2 = self._lookahead_buffer.find("<function_calls>")
                start_idx = tag_idx1 if tag_idx1 != -1 else tag_idx2

                preamble = self._lookahead_buffer[:start_idx]
                if preamble:
                    logger.debug(
                        f"KimiToolCallInterceptor: Dropping pre-tool-call preamble "
                        f"({len(preamble)} chars) to prevent TTS echo: "
                        f"{repr(preamble[:80])}"
                    )

                self._is_buffering_tool = True
                self._tool_call_buffer = self._lookahead_buffer[start_idx:]
                self._lookahead_buffer = ""
                logger.debug(f"KimiToolCallInterceptor: Triggered, buffering tool call stream.")
                return

            # Check for partial tag prefixes — hold them back but flush confirmed-
            # safe text before them at sentence boundaries.
            last_lt = self._lookahead_buffer.rfind('<')
            if last_lt != -1:
                possible_tag = self._lookahead_buffer[last_lt:]
                if len(possible_tag) < 30 and (
                    "<|tool_calls_section_begin|>".startswith(possible_tag) or
                    "<function_calls>".startswith(possible_tag)
                ):
                    # Text before last_lt is safe. Flush it only at a sentence boundary
                    # so we don't hold normal speech too long.
                    safe_region = self._lookahead_buffer[:last_lt]
                    flush_idx = _last_sentence_boundary(safe_region)
                    if flush_idx > 0:
                        to_flush = safe_region[:flush_idx]
                        remainder = safe_region[flush_idx:]
                        if to_flush.strip():
                            await self.push_frame(TextFrame(to_flush), direction)
                        self._lookahead_buffer = remainder + possible_tag
                    # else: keep holding — no sentence boundary before the partial tag yet
                    return

            # No tool call tags or partial starts. Flush at sentence boundaries.
            flush_idx = _last_sentence_boundary(self._lookahead_buffer)
            if flush_idx > 0:
                to_flush = self._lookahead_buffer[:flush_idx]
                self._lookahead_buffer = self._lookahead_buffer[flush_idx:]
                if to_flush.strip():
                    await self.push_frame(TextFrame(to_flush), direction)
            elif len(self._lookahead_buffer) > _MAX_LOOKAHEAD_CHARS:
                # Safety valve: force-flush if buffer is very large with no sentence boundary.
                logger.debug(
                    f"KimiToolCallInterceptor: Force-flushing oversized lookahead buffer "
                    f"({len(self._lookahead_buffer)} chars)"
                )
                if self._lookahead_buffer.strip():
                    await self.push_frame(TextFrame(self._lookahead_buffer), direction)
                self._lookahead_buffer = ""

            return

        else:
            # Non-Text frames (or upstream frames) pass through.
            # For LLMFullResponseEndFrame: flush any remaining lookahead buffer.
            from pipecat.frames.frames import LLMFullResponseEndFrame
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
                # Groq/vLLM can occasionally truncate or omit section-end tags.
                # Finalize any buffered tool payload at end-of-response.
                if self._is_buffering_tool and self._tool_call_buffer:
                    logger.debug("KimiToolCallInterceptor: Finalizing buffered tool payload on response end")
                    tool_calls = extract_tool_call_info(self._tool_call_buffer)
                    if tool_calls and self._llm and hasattr(self._llm, 'run_function_calls'):
                        try:
                            calls_to_run = []
                            for tc in tool_calls:
                                tool_name = tc.get("function", {}).get("name", "")
                                args_raw = tc.get("function", {}).get("arguments", "{}")
                                tool_args = json.loads(args_raw) if args_raw else {}
                                tool_id = tc.get("id", "")
                                calls_to_run.append(FunctionCallFromLLM(
                                    context=self._latest_context,
                                    tool_call_id=tool_id,
                                    function_name=tool_name,
                                    arguments=tool_args
                                ))
                            await self.push_frame(FunctionCallsFromLLMInfoFrame(function_calls=calls_to_run), direction)
                            await self._llm.run_function_calls(calls_to_run)
                        except Exception as e:
                            logger.error(f"KimiToolCallInterceptor: Failed to finalize run_function_calls: {e}")
                    else:
                        logger.debug("KimiToolCallInterceptor: No parsable tool calls found in final buffered payload")
                    self._is_buffering_tool = False
                    self._tool_call_buffer = ""

                # Flush any remaining lookahead text (tail of a normal response)
                if self._lookahead_buffer:
                    if self._lookahead_buffer.strip():
                        await self.push_frame(TextFrame(self._lookahead_buffer), direction)
                    self._lookahead_buffer = ""

            await self.push_frame(frame, direction)
