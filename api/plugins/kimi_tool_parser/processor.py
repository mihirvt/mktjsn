"""Kimi K2 Tool Call Interceptor for Pipecat.

Kimi K2 on DeepInfra / Groq does NOT support standard OpenAI-compatible
``delta.tool_calls`` streaming consistently. It sometimes emits tool calls
as raw XML text inside ``delta.content``, AND it sometimes echoes the user's
last message as a "thinking" preamble before the real answer.

This interceptor sits between the LLM and TTS in the pipeline and:

  1. Buffers the first ``_ECHO_BUFFER_CHARS`` characters of every response.
  2. If those characters contain echoed user words → drops them silently.
  3. Detects ``<function_calls>`` / ``<|tool_calls_section_begin|>`` tags →
     buffers + executes the tool call, then suppresses all trailing text.
  4. Normal text after the echo-guard passes through with minimal latency.

  **No InterruptionFrame** is ever pushed — that freezes the pipeline.
"""

import json
import re
import unicodedata

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

# How many characters to buffer at response start for echo detection.
# Kimi K2's echo block is typically 30-150 chars.
_ECHO_BUFFER_CHARS = 200

# Minimum fraction of echo-buffer chars that must overlap with the user's last
# message for us to classify it as an echo and drop it.
_ECHO_OVERLAP_THRESHOLD = 0.35


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase + strip punctuation for fuzzy matching."""
    text = unicodedata.normalize("NFKC", text.lower())
    return re.sub(r"[\s\W]+", "", text)


def _echo_overlap_ratio(response_start: str, user_message: str) -> float:
    """Calculate how much of response_start overlaps with user_message.

    Uses a sliding-window approach: counts how many 20-char windows from
    the user message appear in the response start. Returns a float 0.0–1.0.
    """
    if not user_message or not response_start:
        return 0.0
    rn = _normalize(response_start)
    un = _normalize(user_message)
    if not rn or not un:
        return 0.0

    # Quick containment: user prefix in response or response prefix in user
    if len(un) >= 12 and un[:12] in rn:
        return 0.85
    if len(rn) >= 12 and rn[:12] in un:
        return 0.85

    # Sliding window: count how many user substrings appear in response start
    window = min(len(un), len(rn), 20)
    if window < 6:
        return 0.0
    total_windows = max(1, len(un) - window + 1)
    hits = 0
    for i in range(0, len(un) - window + 1):
        if un[i : i + window] in rn:
            hits += 1
    return hits / total_windows


def extract_tool_call_info(tool_call_rsp: str, start_idx: int = 0):
    """Parse Kimi K2 raw tool call tags into structured tool call dicts.

    Supports both Moonshot native format (``<|tool_calls_section_begin|>``)
    and DeepInfra/vLLM fallback format (``<function_calls>``).

    CRITICAL: Kimi K2 requires tool_call IDs in the format
    ``functions.func_name:idx`` where idx is a global counter starting
    at 0 that increments with each function invocation across the entire
    conversation. Using random IDs (like ``call_abc123``) causes Kimi K2
    to "crash" on subsequent turns — it dumps raw tokens into content
    instead of making proper tool calls.

    Args:
        tool_call_rsp: Raw text containing tool call XML/tags.
        start_idx: The starting global counter for tool call IDs.
    """
    tool_calls = []
    idx = start_idx

    # 1. Moonshot native format — IDs are already in correct format
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
            idx += 1

    # 2. DeepInfra / vLLM fallback format — generate proper Kimi IDs
    if "<function_calls>" in tool_call_rsp:
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
                    "id": f"functions.{func_name}:{idx}",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps(args)},
                }
            )
            idx += 1

        # <invoke name="func_name"/>  (self-closing, no args)
        self_closing = re.findall(
            r'<invoke\s+name="([^"]+)"\s*/>', block, re.DOTALL
        )
        for func_name in self_closing:
            tool_calls.append(
                {
                    "id": f"functions.{func_name}:{idx}",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps({})},
                }
            )
            idx += 1

    return tool_calls


# ── Interceptor ────────────────────────────────────────────────────────────────

class KimiToolCallInterceptor(FrameProcessor):
    """Zero-latency interceptor for Kimi K2 raw-text tool calls and echo preamble.

    Two separate problems are handled:

    **Problem 1 — XML tool call preamble / trailing text**
      Kimi K2 sometimes outputs:
        ``Some echoed text<function_calls>...</function_calls>More echoed text``
      The interceptor buffers on ``<function_calls>`` detection, executes the
      tool call, then suppresses ALL remaining text in that response.

    **Problem 2 — Non-XML progressive echo ("thinking" bleed-through)**
      Kimi K2 on vLLM sometimes outputs the user's last message progressively
      at the start of its response before giving a real answer:
        ``Acne के लिएAcne के लिए आप कैसे help कर सकतेAcne के लिए हमारे...``
      The interceptor buffers the first ~200 chars and detects if they
      overlap heavily with the user's last utterance. If so, it drops that
      junk prefix and only passes through the actual answer.
    """

    def __init__(self, llm=None):
        super().__init__()
        self._llm = llm

        # ── Tool call state ──────────────────────────────────────────────────
        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        self._lookahead_buffer = ""
        self._flushed_preamble_tokens = False
        self._latest_context = None

        # Global counter for Kimi K2 tool call IDs (functions.name:idx)
        # CRITICAL: Kimi K2 requires sequential IDs starting at 0.
        self._tool_call_counter = 0

        # After a tool call is finalized, suppress ALL remaining text from
        # the current response to prevent trailing garbage from reaching TTS
        # and being saved to conversation history.
        self._suppress_until_response_end = False

        # ── Echo-preamble guard ──────────────────────────────────────────────
        # Buffer the first _ECHO_BUFFER_CHARS of each response to detect
        # whether the model is echoing the user's last message.
        self._echo_guard_active = True      # True = still in guard phase
        self._echo_buffer = ""             # chars collected so far
        self._echo_dropped = False          # True if we discarded an echo prefix
        self._last_user_message = ""       # last user utterance for comparison

    def _reset_state(self):
        """Reset all interceptor state. Called at start of each LLM response."""
        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        self._lookahead_buffer = ""
        self._flushed_preamble_tokens = False
        self._suppress_until_response_end = False

        # Reset echo guard for new response
        self._echo_guard_active = True
        self._echo_buffer = ""
        self._echo_dropped = False

    # ── Main dispatch ──────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Capture the latest context + last user message
        if direction == FrameDirection.DOWNSTREAM:
            try:
                if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
                    self._latest_context = frame.context
                    # Extract last user message for echo detection
                    msgs = getattr(frame.context, "messages", []) or []
                    for msg in reversed(msgs):
                        role = msg.get("role", "") if isinstance(msg, dict) else ""
                        if role == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                self._last_user_message = content
                            break
            except (NameError, AttributeError):
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

    # ── Echo guard ────────────────────────────────────────────────────────

    async def _run_echo_guard(self, text: str, direction: FrameDirection) -> str:
        """Buffer the start of a response and detect/strip echo preamble.

        Returns the text that should be processed further (may be empty if
        still buffering, or the non-echo suffix if an echo was found/confirmed).
        """
        if not self._echo_guard_active:
            return text  # guard already resolved, pass through

        self._echo_buffer += text

        if len(self._echo_buffer) < _ECHO_BUFFER_CHARS:
            # Still collecting — check if an XML tag appeared (early exit)
            for tag in _TAG_STARTS:
                if tag in self._echo_buffer:
                    # It's a tool call preamble — guard done, pass buffer forward
                    self._echo_guard_active = False
                    buf = self._echo_buffer
                    self._echo_buffer = ""
                    return buf
            # Not enough chars yet — keep buffering, return nothing
            return ""

        # We have enough chars to decide
        self._echo_guard_active = False
        full_buf = self._echo_buffer
        self._echo_buffer = ""

        if not self._last_user_message:
            return full_buf  # no reference to compare — pass through

        overlap = _echo_overlap_ratio(full_buf, self._last_user_message)
        if overlap >= _ECHO_OVERLAP_THRESHOLD:
            # This looks like an echo — find where the real answer starts
            # Strategy: find the first occurrence of the FULL user sentence
            # fragment repeated, then take text after the last repetition.
            normalized_buf = _normalize(full_buf)
            normalized_user = _normalize(self._last_user_message)

            # Find where the echoed fragment ends by looking for the last
            # occurrence of any 10-char user prefix in the buffer
            cutoff = 0
            prefix = normalized_user[:15] if len(normalized_user) >= 15 else normalized_user
            pos = normalized_buf.rfind(prefix)
            if pos != -1:
                # Advance past the last echo occurrence
                # Map back to original text position roughly
                ratio = pos / max(len(normalized_buf), 1)
                cutoff = int(ratio * len(full_buf))
                # Skip to end of what appears to be the last echo
                # Look for next sentence boundary after cutoff
                remainder = full_buf[cutoff:]
                # Find first meaningful punct after cutoff
                m = re.search(r"[।\.\?!,]", remainder)
                if m:
                    cutoff += m.end()

            remainder = full_buf[cutoff:].lstrip()
            logger.info(
                f"KimiToolCallInterceptor: Echo guard DROPPED {cutoff} chars "
                f"(overlap={overlap:.2f}): {repr(full_buf[:80])}"
            )
            self._echo_dropped = True
            return remainder

        logger.debug(
            f"KimiToolCallInterceptor: Echo guard passed "
            f"(overlap={overlap:.2f}): {repr(full_buf[:60])}"
        )
        return full_buf

    # ── Text handling ──────────────────────────────────────────────────────

    async def _handle_text(self, frame: TextFrame, direction: FrameDirection):
        text = frame.text

        # GATE: After a tool call was finalized, drop ALL remaining text
        if self._suppress_until_response_end:
            logger.debug(
                f"KimiToolCallInterceptor: Suppressing post-tool-call text "
                f"({len(text)} chars): {repr(text[:60])}"
            )
            return

        # Phase 1: already inside a tool-call XML block — keep buffering
        if self._is_buffering_tool:
            self._tool_call_buffer += text
            if _MOON_TAG_END in self._tool_call_buffer or _VLLM_TAG_END in self._tool_call_buffer:
                await self._finalize_tool_call(direction)
            return

        # ── Echo guard phase ───────────────────────────────────────────────
        text = await self._run_echo_guard(text, direction)
        if not text:
            return  # still buffering the echo guard window

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
            f"parsing... (tool_call_counter={self._tool_call_counter})"
        )
        tool_calls = extract_tool_call_info(
            self._tool_call_buffer, start_idx=self._tool_call_counter
        )

        # Activate suppression gate BEFORE running tool calls so any trailing
        # content tokens are silently dropped.
        self._suppress_until_response_end = True
        self._is_buffering_tool = False
        self._tool_call_buffer = ""
        self._flushed_preamble_tokens = False
        logger.info(
            "KimiToolCallInterceptor: Suppression gate ACTIVATED — "
            "all remaining text from this response will be dropped"
        )

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
                self._tool_call_counter += len(tool_calls)
                logger.info(
                    f"KimiToolCallInterceptor: Submitting {len(calls_to_run)} "
                    f"function call(s): {[(c.function_name, c.tool_call_id) for c in calls_to_run]}"
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

    # ── Response-end handling ──────────────────────────────────────────────

    async def _handle_response_end(self, frame: Frame, direction: FrameDirection):
        """Handle LLMFullResponseEndFrame: finalize any pending tool buffer."""
        if self._is_buffering_tool and self._tool_call_buffer:
            logger.debug(
                "KimiToolCallInterceptor: Finalizing buffered tool payload on response end"
            )
            await self._finalize_tool_call(direction)

        # Flush any remaining echo guard buffer (response ended mid-window)
        if self._echo_guard_active and self._echo_buffer:
            self._echo_guard_active = False
            remaining = self._echo_buffer
            self._echo_buffer = ""
            if remaining.strip() and not self._suppress_until_response_end:
                await self.push_frame(TextFrame(remaining), direction)

        if self._suppress_until_response_end:
            logger.info(
                "KimiToolCallInterceptor: Response ended. Suppression gate deactivated."
            )

        # Flush remaining lookahead (only if not suppressing)
        if not self._suppress_until_response_end and self._lookahead_buffer:
            if self._lookahead_buffer.strip():
                await self.push_frame(TextFrame(self._lookahead_buffer), direction)
            self._lookahead_buffer = ""

        self._flushed_preamble_tokens = False
        self._suppress_until_response_end = False
        await self.push_frame(frame, direction)
