"""Kimi K2 Tool Call Interceptor for Pipecat.

Kimi K2 on DeepInfra / Groq does NOT support standard OpenAI-compatible
``delta.tool_calls`` streaming consistently. It sometimes emits tool calls
as raw XML text inside ``delta.content``.  This interceptor sits between
the LLM and TTS in the pipeline and:

  1. Passes normal text through with **zero latency** (no buffering).
  2. Detects ``<function_calls>`` (vLLM format) or ``<|tool_calls_section_begin|>``
     (Moonshot native format) in the text stream.
  3. When a tool call tag is detected:
       a. Drops the "preamble" text (echo of user words the model generates
          before the tag).
       b. Buffers the tool call XML until the closing tag.
       c. Parses and executes the tool call via the LLM's ``run_function_calls``.

  **No InterruptionFrame is pushed** to avoid freezing the pipeline state.
  Preamble tokens that were already flushed before the tag was detected
  will reach TTS — this is the minimal acceptable audio leak (~50-200ms).
"""

import json
import re

from loguru import logger

from pipecat.frames.frames import (
    TextFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMContextFrame,
    FunctionCallsFromLLMInfoFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
    from pipecat.services.llm_service import FunctionCallFromLLM
except ImportError:
    pass


# ── Tag constants ──────────────────────────────────────────────────────────────

# vLLM / DeepInfra format
_VLLM_TAG_START = "<function_calls>"
_VLLM_TAG_END = "</function_calls>"

# Moonshot native format
_MOON_TAG_START = "<|tool_calls_section_begin|>"
_MOON_TAG_END = "<|tool_calls_section_end|>"

# Union of start-tag prefixes for partial detection
_TAG_STARTS = [_VLLM_TAG_START, _MOON_TAG_START]

# Maximum partial-tag prefix to hold (longest start tag + margin)
_MAX_PARTIAL_TAG_LEN = max(len(t) for t in _TAG_STARTS) + 2


# ── Parsing helpers ────────────────────────────────────────────────────────────

def extract_tool_call_info(tool_call_rsp: str):
    """Parse Kimi K2 raw tool call tags into structured tool call dicts.

    Supports both Moonshot native format (``<|tool_calls_section_begin|>``)
    and DeepInfra/vLLM fallback format (``<function_calls>``).
    """
    tool_calls = []

    # 1. Moonshot native format
    if "<|tool_call_begin|>" in tool_call_rsp:
        func_call_pattern = (
            r"\<\|tool_call_begin\|\>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*"
            r"\<\|tool_call_argument_begin\|\>\s*(?P<function_arguments>.*?)\s*"
            r"\<\|tool_call_end\|\>"
        )
        for match in re.findall(func_call_pattern, tool_call_rsp, re.DOTALL):
            function_id, function_args = match
            function_name = function_id.split(".")[1].split(":")[0]
            tool_calls.append(
                {
                    "id": function_id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": function_args},
                }
            )

    # 2. DeepInfra / vLLM fallback format
    if "<function_calls>" in tool_call_rsp:
        import uuid

        # Try wrapped section first
        sections = re.findall(
            r"<function_calls>(.*?)</function_calls>", tool_call_rsp, re.DOTALL
        )
        block = sections[0] if sections else tool_call_rsp

        # <invoke name="func_name">args</invoke>
        invokes = re.findall(
            r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', block, re.DOTALL
        )
        for func_name, args_block in invokes:
            args = {}
            if args_block and args_block.strip():
                arg_matches = re.findall(r"<([^>]+)>(.*?)</\1>", args_block, re.DOTALL)
                for k, v in arg_matches:
                    args[k] = v.strip()
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps(args)},
                }
            )

        # <invoke name="func_name"/>  (self-closing, no args)
        self_closing = re.findall(
            r'<invoke\s+name="([^"]+)"\s*/>', block, re.DOTALL
        )
        for func_name in self_closing:
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps({})},
                }
            )

    return tool_calls


# ── Interceptor ────────────────────────────────────────────────────────────────

class KimiToolCallInterceptor(FrameProcessor):
    """Zero-latency interceptor for Kimi K2 raw-text tool calls.

    Design:
      * Normal text passes through **immediately** — no buffering.
      * Only holds back a short ``<…`` partial when it might be a tag start.
      * When a full tag is confirmed the preamble is dropped and the tool
        call XML is buffered + parsed + executed.
      * **No InterruptionFrame** is ever pushed. This is critical — pushing
        an InterruptionFrame from inside the pipeline between LLM and TTS
        causes the pipeline to enter a fully-interrupted state and the bot
        freezes (won't respond to user after the interruption).
    """

    def __init__(self, llm=None):
        super().__init__()
        self._llm = llm
        # True once we've detected a tool-call start tag and are collecting XML
        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        # Small lookahead used ONLY for partial-tag detection
        self._lookahead_buffer = ""
        # Track whether any preamble tokens were already flushed downstream
        self._flushed_preamble_tokens = False
        # Latest LLM context for constructing FunctionCallFromLLM
        self._latest_context = None

    def _reset_state(self):
        """Reset all interceptor state. Called at start of each LLM response."""
        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        self._lookahead_buffer = ""
        self._flushed_preamble_tokens = False

    # ── Main dispatch ──────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Capture the latest context so tool calls can reference it
        if direction == FrameDirection.DOWNSTREAM:
            try:
                if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
                    self._latest_context = frame.context
            except NameError:
                pass

        # Reset state at the start of each new LLM response turn
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMFullResponseStartFrame)
        ):
            self._reset_state()
            await self.push_frame(frame, direction)
            return

        # Only intercept downstream TextFrames
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            await self._handle_text(frame, direction)
            return

        # LLMFullResponseEndFrame — finalize any pending tool buffer
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMFullResponseEndFrame)
        ):
            await self._handle_response_end(frame, direction)
            return

        # Everything else passes through
        await self.push_frame(frame, direction)

    # ── Text handling ──────────────────────────────────────────────────────

    async def _handle_text(self, frame: TextFrame, direction: FrameDirection):
        text = frame.text

        # Phase 1: already inside a tool-call XML block — keep buffering
        if self._is_buffering_tool:
            self._tool_call_buffer += text
            if _MOON_TAG_END in self._tool_call_buffer or _VLLM_TAG_END in self._tool_call_buffer:
                await self._finalize_tool_call(direction)
            return

        # Phase 2: normal text — check for tool-call tag start
        self._lookahead_buffer += text

        # Check for a complete start tag
        has_moon = _MOON_TAG_START in self._lookahead_buffer
        has_vllm = _VLLM_TAG_START in self._lookahead_buffer
        if has_moon or has_vllm:
            tag = _MOON_TAG_START if has_moon else _VLLM_TAG_START
            end_tag = _MOON_TAG_END if has_moon else _VLLM_TAG_END
            start_idx = self._lookahead_buffer.find(tag)

            preamble = self._lookahead_buffer[:start_idx]

            if preamble or self._flushed_preamble_tokens:
                logger.debug(
                    f"KimiToolCallInterceptor: Dropping pre-tool-call preamble "
                    f"({len(preamble)} chars in buffer, "
                    f"flushed_earlier={self._flushed_preamble_tokens}): "
                    f"{repr(preamble[:80])}"
                )

            self._is_buffering_tool = True
            self._tool_call_buffer = self._lookahead_buffer[start_idx:]
            self._lookahead_buffer = ""
            self._flushed_preamble_tokens = False
            logger.debug(
                f"KimiToolCallInterceptor: Triggered, buffering tool call stream. "
                f"Buffer so far: {len(self._tool_call_buffer)} chars"
            )

            # Check if the ENTIRE tool call (including end tag) arrived in one frame
            if end_tag in self._tool_call_buffer:
                logger.debug(
                    "KimiToolCallInterceptor: Complete tool call in single frame, "
                    "finalizing immediately."
                )
                await self._finalize_tool_call(direction)
            return

        # Check for a partial tag start — hold back only the '<' or '<|' portion
        last_lt = self._lookahead_buffer.rfind("<")
        if last_lt != -1:
            possible_partial = self._lookahead_buffer[last_lt:]
            if len(possible_partial) < _MAX_PARTIAL_TAG_LEN and any(
                tag.startswith(possible_partial) for tag in _TAG_STARTS
            ):
                # Flush everything BEFORE the partial tag immediately
                safe = self._lookahead_buffer[:last_lt]
                if safe:
                    self._flushed_preamble_tokens = True
                    await self.push_frame(TextFrame(safe), direction)
                self._lookahead_buffer = possible_partial
                return

        # No tag speculation — flush everything (zero latency path)
        if self._lookahead_buffer:
            if self._lookahead_buffer.strip():
                self._flushed_preamble_tokens = True
                await self.push_frame(TextFrame(self._lookahead_buffer), direction)
            self._lookahead_buffer = ""

    # ── Tool call finalization ─────────────────────────────────────────────

    async def _finalize_tool_call(self, direction: FrameDirection):
        """Parse the buffered tool-call XML and execute via the LLM service."""
        logger.debug(
            f"KimiToolCallInterceptor: Buffer complete ({len(self._tool_call_buffer)} chars), "
            f"parsing..."
        )
        tool_calls = extract_tool_call_info(self._tool_call_buffer)

        if tool_calls and self._llm and hasattr(self._llm, "run_function_calls"):
            try:
                calls_to_run = []
                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    args_raw = tc.get("function", {}).get("arguments", "{}")
                    tool_args = json.loads(args_raw) if args_raw else {}
                    tool_id = tc.get("id", "")
                    calls_to_run.append(
                        FunctionCallFromLLM(
                            context=self._latest_context,
                            tool_call_id=tool_id,
                            function_name=tool_name,
                            arguments=tool_args,
                        )
                    )
                logger.info(
                    f"KimiToolCallInterceptor: Submitting {len(calls_to_run)} "
                    f"function call(s): {[c.function_name for c in calls_to_run]}"
                )
                await self.push_frame(
                    FunctionCallsFromLLMInfoFrame(function_calls=calls_to_run),
                    direction,
                )
                await self._llm.run_function_calls(calls_to_run)
            except Exception as e:
                logger.error(f"KimiToolCallInterceptor: Failed to run function calls: {e}")
        elif tool_calls:
            logger.error(
                "KimiToolCallInterceptor: LLM reference missing or lacks run_function_calls"
            )
        else:
            logger.warning(
                f"KimiToolCallInterceptor: Failed to parse buffered tool call: "
                f"{self._tool_call_buffer[:200]}"
            )

        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        self._flushed_preamble_tokens = False

    # ── Response-end handling ──────────────────────────────────────────────

    async def _handle_response_end(self, frame: Frame, direction: FrameDirection):
        """Handle LLMFullResponseEndFrame: finalize any pending tool buffer."""
        if self._is_buffering_tool and self._tool_call_buffer:
            logger.debug(
                "KimiToolCallInterceptor: Finalizing buffered tool payload on response end"
            )
            await self._finalize_tool_call(direction)

        # Flush any remaining lookahead text (e.g. trailing text with no tag)
        if self._lookahead_buffer:
            if self._lookahead_buffer.strip():
                await self.push_frame(TextFrame(self._lookahead_buffer), direction)
            self._lookahead_buffer = ""

        self._flushed_preamble_tokens = False
        await self.push_frame(frame, direction)
