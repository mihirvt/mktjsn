"""
Inworld AI Streaming WebSocket TTS service for Pipecat.

Protocol reference (bidirectional WebSocket):
  Endpoint:  wss://api.inworld.ai/tts/v1/voice:streamBidirectional
  Auth:      authorization=Basic <key>  (query param on upgrade)

  Client → Server:
    Create context:   {"create": {voiceId, modelId, audioConfig, ...}, "contextId": "..."}
    Send text:        {"send_text": {"text": "...", "flush_context": {}}, "contextId": "..."}
    Flush context:    {"flush_context": {}, "contextId": "..."}
    Close context:    {"close_context": {}, "contextId": "..."}

  Server → Client:
    Context created:  {"result": {"contextId": "...", "contextCreated": {...}, "status": {...}}}
    Audio chunk:      {"result": {"contextId": "...", "audioChunk": {"audioContent": "<b64>", ...}}}
    Flush completed:  {"result": {"contextId": "...", "flushCompleted": {}, "status": {...}}}
    Context closed:   {"result": {"contextId": "...", "contextClosed": {}, "status": {...}}}

  Multi-utterance: connection stays open. Each run_tts call creates a new
  context, sends text, waits for audio, then closes that context before the
  next run_tts can proceed.
"""

import asyncio
import base64
import json
import uuid
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from pipecat.services.tts_service import TTSService
except ImportError:
    from pipecat.services.ai_services import TTSService


# ---------------------------------------------------------------------------
# Codec / alignment helpers
# ---------------------------------------------------------------------------

_BYTES_PER_SAMPLE = {
    "PCM": 2,       # 16-bit signed LE, no header
    "LINEAR16": 2,  # 16-bit signed LE + WAV header per chunk
    "WAV": 2,       # 16-bit signed LE + WAV header first chunk
    "MULAW": 1,
    "ALAW": 1,
    "MP3": 1,       # compressed — no alignment needed
    "OGG_OPUS": 1,  # compressed — no alignment needed
}

# LINEAR16 WAV header is 44 bytes, sent per chunk
_WAV_HEADER_SIZE = 44


class InworldTTSService(TTSService):
    """
    Streaming text-to-speech service using Inworld AI bidirectional WebSocket.

    Supports PCM, LINEAR16, WAV, MULAW, ALAW, MP3, and OGG_OPUS encodings.
    For the Pipecat pipeline (telephony / web) we default to PCM (raw 16-bit
    signed little-endian) which avoids WAV header stripping and is compatible
    with all downstream transports.

    IMPORTANT: Always use PCM for Vobiz/telephony transports. WAV and LINEAR16
    embed RIFF headers that cause static noise if not perfectly stripped.
    """

    class InputParams:
        def __init__(
            self,
            *,
            temperature: float = 1.1,
            speaking_rate: float = 1.0,
            auto_mode: bool = True,
            buffer_char_threshold: int = 100,
            max_buffer_delay_ms: int = 0,
        ):
            self.temperature = temperature
            self.speaking_rate = speaking_rate
            self.auto_mode = auto_mode
            self.buffer_char_threshold = buffer_char_threshold
            self.max_buffer_delay_ms = max_buffer_delay_ms

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "Dennis",
        model_id: str = "inworld-tts-1.5-max",
        audio_encoding: str = "PCM",
        sample_rate: int = 24000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._audio_encoding = audio_encoding
        self._sample_rate = sample_rate
        self._params = params or InworldTTSService.InputParams()

        # WebSocket connection state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None

        # Context lifecycle — serializes access so only ONE context is active
        # at a time. _context_finished is cleared when a context starts and
        # set when its contextClosed message arrives.
        self._context_finished = asyncio.Event()
        self._context_finished.set()
        self._context_ready = asyncio.Event()        # set once contextCreated arrives
        self._context_creation_failed = False         # set when context creation returns error

        # Mapping: Inworld contextId → pipeline context_id
        # This is the critical mapping that prevents audio from one Inworld
        # context from being attributed to the wrong pipeline context.
        self._ctx_map: dict[str, str] = {}

        # Track which Inworld contexts have been cleared (interrupted)
        self._cleared_inworld_ctxs: set[str] = set()

        # Track which pipeline contexts have received TTSStartedFrame
        self._started_pipeline_ctxs: set[str] = set()

        # Audio alignment buffers keyed by Inworld context ID
        self._ctx_audio_buffers: dict[str, bytearray] = {}

        # Track first audio chunk per Inworld context (for WAV header stripping)
        self._first_chunk_for_ctx: set[str] = set()

        # The currently active Inworld context ID (the one run_tts is using)
        self._active_inworld_ctx: Optional[str] = None

        # Watchdog
        self._context_watchdog_task: Optional[asyncio.Task] = None
        self._context_timeout_secs = 30.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartInterruptionFrame):
            logger.debug("Inworld TTS received StartInterruptionFrame (legacy)")
        await super().process_frame(frame, direction)

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        logger.debug(
            f"Inworld TTS handling interruption frame={frame.__class__.__name__} "
            f"direction={direction} active_inworld_ctx={self._active_inworld_ctx}"
        )
        await super()._handle_interruption(frame, direction)
        await self._clear_active_context()

    # ------------------------------------------------------------------
    # run_tts – main entry point called by Pipecat pipeline
    # ------------------------------------------------------------------

    @traced_tts
    async def run_tts(
        self, text: str, context_id: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[Frame, None]:
        if not self._has_speakable_content(text):
            return

        await self._connect()
        if self._ws is None or self._ws.closed:
            yield ErrorFrame("Inworld TTS WebSocket not connected")
            return

        turn_pipeline_ctx = context_id or f"inworld-{uuid.uuid4()}"

        # ---- Persistent context: reuse existing or create new ----
        # Instead of creating/closing a context per sentence (which adds
        # ~200-400ms overhead per sentence and causes choppy transitions),
        # we keep ONE context alive for the entire turn. With autoMode on,
        # the Inworld server handles buffering and flushing internally.
        # Context is only closed on interruption or session end.
        if self._active_inworld_ctx is None:
            # Wait for any previous context to fully close (e.g. from interruption)
            try:
                await asyncio.wait_for(
                    self._context_finished.wait(),
                    timeout=self._context_timeout_secs,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Inworld TTS: previous context did not finish within "
                    f"{self._context_timeout_secs}s — force-clearing"
                )
                await self._clear_active_context()

            # Create a new Inworld context
            inworld_ctx = f"ctx-{uuid.uuid4().hex[:12]}"

            self._context_finished.clear()
            self._ctx_map[inworld_ctx] = turn_pipeline_ctx
            self._active_inworld_ctx = inworld_ctx
            self._ctx_audio_buffers[inworld_ctx] = bytearray()
            self._started_pipeline_ctxs.add(turn_pipeline_ctx)
            self._first_chunk_for_ctx.add(inworld_ctx)
            self._context_ready.clear()
            self._context_creation_failed = False
            self._schedule_context_watchdog(inworld_ctx)

            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame(context_id=turn_pipeline_ctx)

            create_msg = {
                "create": {
                    "voiceId": self._voice_id,
                    "modelId": self._model_id,
                    "audioConfig": {
                        "audioEncoding": self._audio_encoding,
                        "sampleRateHertz": int(self._sample_rate),
                        "speakingRate": round(float(self._params.speaking_rate), 2),
                    },
                    "temperature": round(float(self._params.temperature), 2),
                    "autoMode": bool(self._params.auto_mode),
                    "bufferCharThreshold": int(self._params.buffer_char_threshold),
                },
                "contextId": inworld_ctx,
            }
            if self._params.max_buffer_delay_ms > 0:
                create_msg["create"]["maxBufferDelayMs"] = int(
                    self._params.max_buffer_delay_ms
                )

            await self._send_json(create_msg)

            # Wait for contextCreated confirmation (with timeout)
            try:
                await asyncio.wait_for(self._context_ready.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Inworld TTS: Timed out waiting for contextCreated")
                yield ErrorFrame("Inworld TTS: context creation timed out")
                await self._clear_active_context()
                return

            if self._context_creation_failed:
                logger.warning(
                    f"Inworld TTS: context creation failed for ctx={inworld_ctx}, "
                    f"aborting text send"
                )
                await self._clear_active_context()
                return
        else:
            # Reuse existing context — just update the pipeline mapping
            inworld_ctx = self._active_inworld_ctx
            self._ctx_map[inworld_ctx] = turn_pipeline_ctx
            if turn_pipeline_ctx not in self._started_pipeline_ctxs:
                self._started_pipeline_ctxs.add(turn_pipeline_ctx)
                yield TTSStartedFrame(context_id=turn_pipeline_ctx)
            await self.start_tts_usage_metrics(text)
            self._schedule_context_watchdog(inworld_ctx)

        # Send text — with autoMode on, the server handles flush timing.
        # Inworld has a 1000-char limit per send_text. Chunk if needed.
        remaining = text
        while remaining:
            chunk = remaining[:1000]
            remaining = remaining[1000:]

            send_msg = {
                "send_text": {
                    "text": chunk,
                },
                "contextId": inworld_ctx,
            }
            await self._send_json(send_msg)

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        if self._ws is not None and not self._ws.closed:
            return

        if self._session is None:
            self._session = aiohttp.ClientSession()

        ws_url = (
            f"wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
            f"?authorization=Basic {self._api_key}"
        )

        try:
            logger.debug(
                f"Inworld TTS connecting: voice={self._voice_id} model={self._model_id} "
                f"encoding={self._audio_encoding} sample_rate={self._sample_rate}"
            )
            self._ws = await self._session.ws_connect(
                ws_url,
                heartbeat=20.0,
                autoping=True,
                max_msg_size=0,  # no limit on incoming messages
            )
            self._receive_task = asyncio.create_task(self._receive_audio())
            logger.info("Inworld TTS WebSocket connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Inworld TTS: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Inworld TTS Connection Error: {e}"))

    async def _disconnect(self):
        await self._cancel_context_watchdog()

        # Try to close any active Inworld context gracefully
        if self._active_inworld_ctx and self._ws and not self._ws.closed:
            try:
                await self._send_json({
                    "close_context": {},
                    "contextId": self._active_inworld_ctx,
                })
            except Exception:
                pass  # best-effort

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        self._active_inworld_ctx = None
        self._ctx_map.clear()
        self._cleared_inworld_ctxs.clear()
        self._started_pipeline_ctxs.clear()
        self._ctx_audio_buffers.clear()
        self._first_chunk_for_ctx.clear()
        self._context_finished.set()
        self._context_creation_failed = False
        self._context_ready.set()

    async def _send_json(self, payload: dict):
        await self._connect()
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Inworld TTS WebSocket disconnected")
        try:
            logger.debug(f"Inworld TTS → {json.dumps(payload)[:400]}")
            await self._ws.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send payload to Inworld TTS: {e}", exc_info=True)
            await self._handle_socket_failure(
                f"Inworld TTS Request Error: {e}", from_receive_task=False
            )
            await self.push_frame(ErrorFrame(f"Inworld TTS Request Error: {e}"))

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_audio(self):
        try:
            while self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_text_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"Inworld TTS WebSocket closed: {msg.type}")
                    await self._handle_socket_failure(
                        f"Inworld TTS WebSocket closed: {msg.type}",
                        from_receive_task=True,
                    )
                    break
        except asyncio.CancelledError:
            logger.debug("Inworld TTS receive task cancelled")
        except Exception as e:
            logger.error(f"Inworld TTS stream error: {e}", exc_info=True)
            await self._handle_socket_failure(
                f"Inworld TTS Stream Error: {e}", from_receive_task=True
            )
            await self.push_frame(ErrorFrame(f"Inworld TTS Stream Error: {e}"))

    async def _handle_text_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Inworld TTS non-JSON payload: {raw[:200]}")
            return

        result = data.get("result")
        if not result:
            logger.debug(f"Inworld TTS unexpected message (no result): {raw[:300]}")
            return

        inworld_ctx = result.get("contextId")
        # CRITICAL: Look up the pipeline context via the mapping, NOT from
        # self._active_inworld_ctx. This prevents audio from one Inworld
        # context being attributed to a different pipeline context.
        pipeline_ctx = self._ctx_map.get(inworld_ctx) if inworld_ctx else None
        status = result.get("status", {})

        # Check for errors in status
        if status.get("code", 0) != 0:
            error_msg = status.get("message", "Unknown Inworld TTS error")
            logger.error(
                f"Inworld TTS API error: code={status.get('code')} "
                f"message={error_msg} inworld_ctx={inworld_ctx}"
            )
            # Mark creation as failed so run_tts won't try to send text
            self._context_creation_failed = True
            self._context_ready.set()  # unblock the waiter
            # Clean up this specific context
            if pipeline_ctx and pipeline_ctx in self._started_pipeline_ctxs:
                self._started_pipeline_ctxs.discard(pipeline_ctx)
                await self.push_frame(TTSStoppedFrame(context_id=pipeline_ctx))
            # Only set _context_finished if this is the active context
            if inworld_ctx and inworld_ctx == self._active_inworld_ctx:
                self._context_finished.set()
                await self._cancel_context_watchdog()
            await self.push_frame(ErrorFrame(f"Inworld TTS Error: {error_msg}"))
            return

        # Drop audio for cleared (interrupted) contexts
        if inworld_ctx and inworld_ctx in self._cleared_inworld_ctxs:
            logger.debug(f"Dropping message for cleared Inworld ctx {inworld_ctx}")
            # Still handle contextClosed for cleanup
            if "contextClosed" in result:
                self._cleanup_inworld_ctx(inworld_ctx)
            return

        # --- contextCreated ---
        if "contextCreated" in result:
            logger.debug(
                f"Inworld TTS context created: ctx={inworld_ctx} "
                f"voice={result['contextCreated'].get('voiceId')} "
                f"encoding={result['contextCreated'].get('audioConfig', {}).get('audioEncoding')}"
            )
            self._context_ready.set()
            return

        # --- audioChunk ---
        if "audioChunk" in result:
            audio_chunk = result["audioChunk"]
            audio_b64 = audio_chunk.get("audioContent", "")
            if not audio_b64:
                return
            audio_bytes = base64.b64decode(audio_b64)
            if not audio_bytes:
                return

            # Strip WAV/RIFF headers for LINEAR16/WAV encodings
            audio_bytes = self._strip_wav_header_if_needed(inworld_ctx, audio_bytes)

            await self.stop_ttfb_metrics()
            if inworld_ctx:
                self._schedule_context_watchdog(inworld_ctx)

            audio_bytes = self._buffer_aligned_audio(inworld_ctx, audio_bytes)
            if not audio_bytes:
                return

            frame_kwargs = {
                "audio": audio_bytes,
                "sample_rate": self._sample_rate,
                "num_channels": 1,
            }
            if pipeline_ctx:
                frame_kwargs["context_id"] = pipeline_ctx
            try:
                frame = TTSAudioRawFrame(**frame_kwargs)
            except TypeError:
                frame = TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
            await self.push_frame(frame)
            return

        # --- flushCompleted ---
        if "flushCompleted" in result:
            logger.debug(f"Inworld TTS flush completed for ctx {inworld_ctx}")
            # With persistent context + autoMode, flushCompleted indicates a
            # generation batch finished. The context stays open for more text.
            # Push TTSStoppedFrame to signal this generation is done, but
            # keep the Inworld context alive for reuse.
            if pipeline_ctx and pipeline_ctx in self._started_pipeline_ctxs:
                self._started_pipeline_ctxs.discard(pipeline_ctx)
                await self.push_frame(TTSStoppedFrame(context_id=pipeline_ctx))
            await self.stop_ttfb_metrics()
            return

        # --- contextClosed ---
        if "contextClosed" in result:
            logger.debug(f"Inworld TTS context closed: {inworld_ctx}")

            # Flush any remaining buffered bytes
            final_audio = self._flush_ctx_buffer(inworld_ctx)
            if final_audio:
                frame_kwargs = {
                    "audio": final_audio,
                    "sample_rate": self._sample_rate,
                    "num_channels": 1,
                }
                if pipeline_ctx:
                    frame_kwargs["context_id"] = pipeline_ctx
                try:
                    frame = TTSAudioRawFrame(**frame_kwargs)
                except TypeError:
                    frame = TTSAudioRawFrame(
                        audio=final_audio,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                await self.push_frame(frame)

            # Push TTSStoppedFrame for this pipeline context
            if pipeline_ctx and pipeline_ctx in self._started_pipeline_ctxs:
                self._started_pipeline_ctxs.discard(pipeline_ctx)
                await self.push_frame(TTSStoppedFrame(context_id=pipeline_ctx))
            elif not pipeline_ctx:
                await self.push_frame(TTSStoppedFrame())

            # Mark context as finished if it's the active one
            if inworld_ctx == self._active_inworld_ctx:
                self._active_inworld_ctx = None
                self._context_finished.set()
                await self._cancel_context_watchdog()

            # Cleanup mapping
            self._cleanup_inworld_ctx(inworld_ctx)
            return

    # ------------------------------------------------------------------
    # Context management (interruptions, watchdogs)
    # ------------------------------------------------------------------

    def _cleanup_inworld_ctx(self, inworld_ctx: str):
        """Remove all traces of an Inworld context from internal state."""
        self._ctx_map.pop(inworld_ctx, None)
        self._ctx_audio_buffers.pop(inworld_ctx, None)
        self._first_chunk_for_ctx.discard(inworld_ctx)
        self._cleared_inworld_ctxs.discard(inworld_ctx)

    async def _clear_active_context(self):
        inworld_ctx = self._active_inworld_ctx
        if not inworld_ctx:
            return

        pipeline_ctx = self._ctx_map.get(inworld_ctx)
        logger.debug(
            f"Inworld TTS clearing active context: "
            f"inworld_ctx={inworld_ctx} pipeline_ctx={pipeline_ctx}"
        )

        # Mark as cleared so receive loop drops further audio
        self._cleared_inworld_ctxs.add(inworld_ctx)

        # Try to close on Inworld's side
        if self._ws and not self._ws.closed:
            try:
                await self._send_json({
                    "close_context": {},
                    "contextId": inworld_ctx,
                })
            except Exception:
                pass

        self._active_inworld_ctx = None
        self._context_finished.set()
        self._context_creation_failed = False
        self._context_ready.set()
        await self._cancel_context_watchdog()

        if pipeline_ctx and pipeline_ctx in self._started_pipeline_ctxs:
            self._started_pipeline_ctxs.discard(pipeline_ctx)
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStoppedFrame(context_id=pipeline_ctx))

    def _schedule_context_watchdog(self, inworld_ctx: str):
        if self._context_watchdog_task:
            self._context_watchdog_task.cancel()
        self._context_watchdog_task = asyncio.create_task(
            self._context_watchdog(inworld_ctx)
        )

    async def _cancel_context_watchdog(self):
        if not self._context_watchdog_task:
            return
        self._context_watchdog_task.cancel()
        try:
            await self._context_watchdog_task
        except asyncio.CancelledError:
            pass
        self._context_watchdog_task = None

    async def _context_watchdog(self, inworld_ctx: str):
        try:
            await asyncio.sleep(self._context_timeout_secs)
            if self._active_inworld_ctx != inworld_ctx:
                return
            logger.warning(
                f"Inworld TTS context {inworld_ctx} timed out after "
                f"{self._context_timeout_secs}s"
            )
            await self._handle_socket_failure(
                f"Inworld TTS context {inworld_ctx} timed out",
                from_receive_task=False,
            )
            await self.push_frame(
                ErrorFrame(f"Inworld TTS context {inworld_ctx} timed out")
            )
        except asyncio.CancelledError:
            raise

    async def _handle_socket_failure(self, reason: str, *, from_receive_task: bool):
        inworld_ctx = self._active_inworld_ctx
        pipeline_ctx = self._ctx_map.get(inworld_ctx) if inworld_ctx else None
        logger.warning(
            f"Inworld TTS transport failure. reason={reason} "
            f"active_inworld_ctx={inworld_ctx} pipeline_ctx={pipeline_ctx}"
        )

        self._active_inworld_ctx = None
        self._context_finished.set()
        self._context_creation_failed = False
        self._context_ready.set()
        self._ctx_map.clear()
        self._cleared_inworld_ctxs.clear()
        self._ctx_audio_buffers.clear()
        self._first_chunk_for_ctx.clear()
        await self._cancel_context_watchdog()

        if pipeline_ctx and pipeline_ctx in self._started_pipeline_ctxs:
            self._started_pipeline_ctxs.discard(pipeline_ctx)
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStoppedFrame(context_id=pipeline_ctx))

        current_task = asyncio.current_task()
        receive_task = self._receive_task
        self._receive_task = None

        if receive_task and not from_receive_task and receive_task is not current_task:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _has_speakable_content(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return any(char.isalnum() for char in stripped)

    def _strip_wav_header_if_needed(
        self, inworld_ctx: Optional[str], audio_bytes: bytes
    ) -> bytes:
        """Strip WAV/RIFF header from audio chunks if encoding produces them.

        - LINEAR16: WAV header on EVERY chunk → strip from all chunks.
        - WAV: WAV header on first chunk of each *event* (including auto-mode
          internal flushes). With autoMode, Inworld may produce multiple events
          within a single context, so we must check ALL chunks for RIFF headers.
        - PCM, MULAW, ALAW, MP3, OGG_OPUS: no header → pass through.
        """
        if self._audio_encoding in ("LINEAR16", "WAV"):
            # Both LINEAR16 and WAV can have RIFF headers — check every chunk.
            # LINEAR16: header on every chunk; WAV: header on first chunk of
            # each generation event (can be multiple per context with autoMode).
            if len(audio_bytes) > _WAV_HEADER_SIZE and audio_bytes[:4] == b"RIFF":
                return audio_bytes[_WAV_HEADER_SIZE:]
            return audio_bytes

        return audio_bytes

    def _buffer_aligned_audio(
        self, inworld_ctx: Optional[str], audio_bytes: bytes
    ) -> bytes:
        """Ensure audio is aligned to sample boundaries (e.g. 2 bytes for PCM16)."""
        alignment = _BYTES_PER_SAMPLE.get(self._audio_encoding, 2)
        if alignment <= 1:
            # No alignment needed for 1-byte codecs (mulaw, alaw, compressed)
            return audio_bytes

        if not inworld_ctx:
            aligned_length = len(audio_bytes) & ~(alignment - 1)
            return audio_bytes[:aligned_length]

        buffer = self._ctx_audio_buffers.setdefault(inworld_ctx, bytearray())
        buffer.extend(audio_bytes)
        aligned_length = len(buffer) & ~(alignment - 1)
        if aligned_length <= 0:
            return b""

        aligned = bytes(buffer[:aligned_length])
        del buffer[:aligned_length]
        return aligned

    def _flush_ctx_buffer(self, inworld_ctx: Optional[str]) -> bytes:
        if not inworld_ctx:
            return b""
        buffer = self._ctx_audio_buffers.pop(inworld_ctx, None)
        if not buffer:
            return b""
        alignment = _BYTES_PER_SAMPLE.get(self._audio_encoding, 2)
        if alignment > 1 and len(buffer) % alignment != 0:
            # Pad to alignment
            buffer.extend(b"\x00" * (alignment - len(buffer) % alignment))
        return bytes(buffer)
