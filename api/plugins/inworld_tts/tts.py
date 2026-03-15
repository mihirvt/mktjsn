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

  Multi-utterance: connection stays open. Close the context after each
  turn completes, then create a new context for the next turn.
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
            apply_text_normalization: str = "APPLY_TEXT_NORMALIZATION_UNSPECIFIED",
        ):
            self.temperature = temperature
            self.speaking_rate = speaking_rate
            self.auto_mode = auto_mode
            self.buffer_char_threshold = buffer_char_threshold
            self.max_buffer_delay_ms = max_buffer_delay_ms
            self.apply_text_normalization = apply_text_normalization

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

        # Context lifecycle
        self._active_context_id: Optional[str] = None
        self._inworld_context_id: Optional[str] = None  # Inworld's context id
        self._started_contexts: set[str] = set()
        self._cleared_contexts: set[str] = set()
        self._context_finished = asyncio.Event()
        self._context_finished.set()
        self._context_ready = asyncio.Event()  # set once contextCreated arrives
        self._context_buffers: dict[str, bytearray] = {}
        self._context_watchdog_task: Optional[asyncio.Task] = None
        self._context_timeout_secs = 30.0
        self._first_chunk_for_context: set[str] = set()  # track first chunks for LINEAR16/WAV header stripping

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
            f"direction={direction} active_context={self._active_context_id}"
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

        # If we still have a dangling previous context, force-finish it
        if not self._context_finished.is_set() and self._active_context_id:
            logger.warning(
                f"Inworld TTS detected stale active context "
                f"{self._active_context_id} — clearing before new synthesis"
            )
            await self._clear_active_context()

        await self._context_finished.wait()
        turn_context_id = context_id or f"inworld-{uuid.uuid4()}"
        inworld_ctx = f"ctx-{uuid.uuid4().hex[:12]}"

        self._context_finished.clear()
        self._active_context_id = turn_context_id
        self._inworld_context_id = inworld_ctx
        self._cleared_contexts.discard(turn_context_id)
        self._context_buffers[turn_context_id] = bytearray()
        self._started_contexts.add(turn_context_id)
        self._first_chunk_for_context.add(inworld_ctx)
        self._context_ready.clear()
        self._schedule_context_watchdog(turn_context_id)

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield TTSStartedFrame(context_id=turn_context_id)

        # 1. Create a context
        create_msg = {
            "create": {
                "voiceId": self._voice_id,
                "modelId": self._model_id,
                "audioConfig": {
                    "audioEncoding": self._audio_encoding,
                    "sampleRateHertz": int(self._sample_rate),
                },
                "temperature": float(self._params.temperature),
                "autoMode": bool(self._params.auto_mode),
                "bufferCharThreshold": int(self._params.buffer_char_threshold),
                "applyTextNormalization": self._params.apply_text_normalization,
            },
            "contextId": inworld_ctx,
        }
        if self._params.speaking_rate != 1.0:
            create_msg["create"]["audioConfig"]["speakingRate"] = float(
                self._params.speaking_rate
            )
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

        # 2. Send text with flush
        # Inworld has a 1000-char limit per send_text. Chunk if needed.
        remaining = text
        while remaining:
            chunk = remaining[:1000]
            remaining = remaining[1000:]
            is_last_chunk = len(remaining) == 0

            send_msg = {
                "send_text": {
                    "text": chunk,
                },
                "contextId": inworld_ctx,
            }
            if is_last_chunk:
                send_msg["send_text"]["flush_context"] = {}

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
        if self._inworld_context_id and self._ws and not self._ws.closed:
            try:
                await self._send_json({
                    "close_context": {},
                    "contextId": self._inworld_context_id,
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

        self._active_context_id = None
        self._inworld_context_id = None
        self._cleared_contexts.clear()
        self._started_contexts.clear()
        self._context_buffers.clear()
        self._first_chunk_for_context.clear()
        self._context_finished.set()
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
        context_id = self._active_context_id
        status = result.get("status", {})

        # Check for errors in status
        if status.get("code", 0) != 0:
            error_msg = status.get("message", "Unknown Inworld TTS error")
            logger.error(f"Inworld TTS API error: code={status.get('code')} message={error_msg}")
            self._context_finished.set()
            self._context_ready.set()
            await self.push_frame(ErrorFrame(f"Inworld TTS Error: {error_msg}"))
            return

        if context_id and context_id in self._cleared_contexts:
            logger.debug(f"Dropping stale Inworld chunk for cleared context {context_id}")
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

            # Strip WAV/RIFF headers for LINEAR16 (header per chunk) or WAV
            # (header on first chunk) so downstream gets clean PCM.
            audio_bytes = self._strip_wav_header_if_needed(inworld_ctx, audio_bytes)

            await self.stop_ttfb_metrics()
            if context_id:
                self._schedule_context_watchdog(context_id)

            audio_bytes = self._buffer_aligned_audio(context_id, audio_bytes)
            if not audio_bytes:
                return

            frame_kwargs = {
                "audio": audio_bytes,
                "sample_rate": self._sample_rate,
                "num_channels": 1,
            }
            if context_id:
                frame_kwargs["context_id"] = context_id
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
            logger.debug(f"Inworld TTS flush completed for context {inworld_ctx}")
            # After flush completes, close the context to release resources
            if inworld_ctx and self._ws and not self._ws.closed:
                try:
                    await self._send_json({
                        "close_context": {},
                        "contextId": inworld_ctx,
                    })
                except Exception as e:
                    logger.warning(f"Inworld TTS failed to close context after flush: {e}")
            return

        # --- contextClosed ---
        if "contextClosed" in result:
            logger.debug(f"Inworld TTS context closed: {inworld_ctx}")

            # Flush any remaining buffered bytes
            final_audio = self._flush_context_buffer(context_id)
            if final_audio:
                frame_kwargs = {
                    "audio": final_audio,
                    "sample_rate": self._sample_rate,
                    "num_channels": 1,
                }
                if context_id:
                    frame_kwargs["context_id"] = context_id
                try:
                    frame = TTSAudioRawFrame(**frame_kwargs)
                except TypeError:
                    frame = TTSAudioRawFrame(
                        audio=final_audio,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                await self.push_frame(frame)

            # Clean up the first-chunk tracker
            if inworld_ctx:
                self._first_chunk_for_context.discard(inworld_ctx)

            if context_id:
                self._cleared_contexts.discard(context_id)
                if context_id in self._started_contexts:
                    self._started_contexts.discard(context_id)
                    await self.push_frame(TTSStoppedFrame(context_id=context_id))
                if context_id == self._active_context_id:
                    self._active_context_id = None
                    self._inworld_context_id = None
                    self._context_finished.set()
                    await self._cancel_context_watchdog()
            else:
                self._active_context_id = None
                self._inworld_context_id = None
                self._context_finished.set()
                await self._cancel_context_watchdog()
                await self.push_frame(TTSStoppedFrame())
            return

    # ------------------------------------------------------------------
    # Context management (interruptions, watchdogs)
    # ------------------------------------------------------------------

    async def _clear_active_context(self):
        context_id = self._active_context_id
        inworld_ctx = self._inworld_context_id
        if not context_id:
            return

        # Try to close on Inworld's side
        if inworld_ctx and self._ws and not self._ws.closed:
            try:
                await self._send_json({
                    "close_context": {},
                    "contextId": inworld_ctx,
                })
            except Exception:
                pass

        self._cleared_contexts.add(context_id)
        self._context_buffers.pop(context_id, None)
        if inworld_ctx:
            self._first_chunk_for_context.discard(inworld_ctx)
        self._active_context_id = None
        self._inworld_context_id = None
        self._context_finished.set()
        self._context_ready.set()
        await self._cancel_context_watchdog()

        if context_id in self._started_contexts:
            self._started_contexts.discard(context_id)
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStoppedFrame(context_id=context_id))

    def _schedule_context_watchdog(self, context_id: str):
        if self._context_watchdog_task:
            self._context_watchdog_task.cancel()
        self._context_watchdog_task = asyncio.create_task(
            self._context_watchdog(context_id)
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

    async def _context_watchdog(self, context_id: str):
        try:
            await asyncio.sleep(self._context_timeout_secs)
            if self._active_context_id != context_id:
                return
            logger.warning(
                f"Inworld TTS context {context_id} timed out after "
                f"{self._context_timeout_secs}s"
            )
            await self._handle_socket_failure(
                f"Inworld TTS context {context_id} timed out",
                from_receive_task=False,
            )
            await self.push_frame(
                ErrorFrame(f"Inworld TTS context {context_id} timed out")
            )
        except asyncio.CancelledError:
            raise

    async def _handle_socket_failure(self, reason: str, *, from_receive_task: bool):
        context_id = self._active_context_id
        logger.warning(
            f"Inworld TTS transport failure. reason={reason} "
            f"active_context={context_id}"
        )

        self._active_context_id = None
        self._inworld_context_id = None
        self._context_finished.set()
        self._context_ready.set()
        self._cleared_contexts.clear()
        self._context_buffers.clear()
        self._first_chunk_for_context.clear()
        await self._cancel_context_watchdog()

        if context_id and context_id in self._started_contexts:
            self._started_contexts.discard(context_id)
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStoppedFrame(context_id=context_id))

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
        - WAV: WAV header on FIRST chunk only (and first after each flush).
        - PCM, MULAW, ALAW, MP3, OGG_OPUS: no header → pass through.
        """
        if self._audio_encoding == "LINEAR16":
            # Every chunk has a 44-byte RIFF header
            if len(audio_bytes) > _WAV_HEADER_SIZE and audio_bytes[:4] == b"RIFF":
                return audio_bytes[_WAV_HEADER_SIZE:]
            return audio_bytes

        if self._audio_encoding == "WAV":
            # Only the first chunk per context has the header
            if inworld_ctx and inworld_ctx in self._first_chunk_for_context:
                self._first_chunk_for_context.discard(inworld_ctx)
                if len(audio_bytes) > _WAV_HEADER_SIZE and audio_bytes[:4] == b"RIFF":
                    return audio_bytes[_WAV_HEADER_SIZE:]
            return audio_bytes

        return audio_bytes

    def _buffer_aligned_audio(
        self, context_id: Optional[str], audio_bytes: bytes
    ) -> bytes:
        """Ensure audio is aligned to sample boundaries (e.g. 2 bytes for PCM16)."""
        alignment = _BYTES_PER_SAMPLE.get(self._audio_encoding, 2)
        if alignment <= 1:
            # No alignment needed for 1-byte codecs (mulaw, alaw, compressed)
            return audio_bytes

        if not context_id:
            aligned_length = len(audio_bytes) & ~(alignment - 1)
            return audio_bytes[:aligned_length]

        buffer = self._context_buffers.setdefault(context_id, bytearray())
        buffer.extend(audio_bytes)
        aligned_length = len(buffer) & ~(alignment - 1)
        if aligned_length <= 0:
            return b""

        aligned = bytes(buffer[:aligned_length])
        del buffer[:aligned_length]
        return aligned

    def _flush_context_buffer(self, context_id: Optional[str]) -> bytes:
        if not context_id:
            return b""
        buffer = self._context_buffers.pop(context_id, None)
        if not buffer:
            return b""
        alignment = _BYTES_PER_SAMPLE.get(self._audio_encoding, 2)
        if alignment > 1 and len(buffer) % alignment != 0:
            # Pad to alignment
            buffer.extend(b"\x00" * (alignment - len(buffer) % alignment))
        return bytes(buffer)
