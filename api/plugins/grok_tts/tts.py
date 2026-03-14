"""
xAI Grok Streaming WebSocket TTS service for Pipecat.

Protocol reference (bidirectional WebSocket):
  Endpoint:  wss://api.x.ai/v1/tts
  Auth:      Authorization: Bearer <key>  (on upgrade)
  Config:    query params — language, voice, codec, sample_rate

  Client → Server:
    {"type": "text.delta", "delta": "partial text …"}
    {"type": "text.done"}

  Server → Client:
    {"type": "audio.delta", "delta": "<base64-encoded audio>"}
    {"type": "audio.done",  "trace_id": "uuid"}
    {"type": "error",       "message": "description"}

  Multi-utterance: connection stays open after audio.done; send next
  text.delta → text.done round without reconnecting.
"""

import asyncio
import base64
import json
import uuid
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

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
# Codec helpers
# ---------------------------------------------------------------------------

# Bytes per sample for aligned-buffer logic
_BYTES_PER_SAMPLE = {
    "pcm": 2,      # 16-bit signed LE
    "mulaw": 1,
    "ulaw": 1,
    "alaw": 1,
}


class GrokTTSService(TTSService):
    """
    Streaming text-to-speech service using xAI Grok bidirectional WebSocket.
    """

    class InputParams:
        def __init__(
            self,
            language: str = "en",
        ):
            self.language = language

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "eve",
        codec: str = "pcm",
        sample_rate: int = 24000,
        language: str = "en",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._voice = voice
        self._codec = codec
        self._sample_rate = sample_rate
        self._language = language
        self._params = params or GrokTTSService.InputParams()

        # WebSocket connection state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None

        # Utterance lifecycle
        self._active_context_id: Optional[str] = None
        self._started_contexts: set[str] = set()
        self._cleared_contexts: set[str] = set()
        self._context_finished = asyncio.Event()
        self._context_finished.set()
        self._context_buffers: dict[str, bytearray] = {}
        self._context_watchdog_task: Optional[asyncio.Task] = None
        self._context_timeout_secs = 30.0  # generous for cold-start

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
            logger.debug("Grok TTS received StartInterruptionFrame (legacy)")
        await super().process_frame(frame, direction)

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        logger.debug(
            f"Grok TTS handling interruption frame={frame.__class__.__name__} "
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
            yield ErrorFrame("Grok TTS WebSocket not connected")
            return

        # If we still have a dangling previous context, force-finish it
        if not self._context_finished.is_set() and self._active_context_id:
            logger.warning(
                f"Grok TTS detected stale active context "
                f"{self._active_context_id} — clearing before new synthesis"
            )
            await self._clear_active_context()

        await self._context_finished.wait()
        turn_context_id = context_id or f"grok-{uuid.uuid4()}"
        self._context_finished.clear()
        self._active_context_id = turn_context_id
        self._cleared_contexts.discard(turn_context_id)
        self._context_buffers[turn_context_id] = bytearray()
        self._started_contexts.add(turn_context_id)
        self._schedule_context_watchdog(turn_context_id)

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield TTSStartedFrame(context_id=turn_context_id)

        # Send text as a single delta + done (complete sentence from LLM aggregator)
        await self._send_json({"type": "text.delta", "delta": text})
        await self._send_json({"type": "text.done"})

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        if self._ws is not None and not self._ws.closed:
            return

        if self._session is None:
            self._session = aiohttp.ClientSession()

        query = urlencode({
            "language": self._params.language or self._language,
            "voice": self._voice,
            "codec": self._codec,
            "sample_rate": str(self._sample_rate),
        })
        ws_url = f"wss://api.x.ai/v1/tts?{query}"

        try:
            logger.debug(
                f"Grok TTS connecting: voice={self._voice} codec={self._codec} "
                f"sample_rate={self._sample_rate}"
            )
            self._ws = await self._session.ws_connect(
                ws_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                heartbeat=20.0,
                autoping=True,
            )
            self._receive_task = asyncio.create_task(self._receive_audio())
            logger.info("Grok TTS WebSocket connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Grok TTS: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Grok TTS Connection Error: {e}"))

    async def _disconnect(self):
        await self._cancel_context_watchdog()
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
        self._cleared_contexts.clear()
        self._started_contexts.clear()
        self._context_buffers.clear()
        self._context_finished.set()

    async def _send_json(self, payload: dict):
        await self._connect()
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Grok TTS WebSocket disconnected")
        try:
            logger.debug(f"Grok TTS → {json.dumps(payload)[:300]}")
            await self._ws.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send payload to Grok TTS: {e}", exc_info=True)
            await self._handle_socket_failure(
                f"Grok TTS Request Error: {e}", from_receive_task=False
            )
            await self.push_frame(ErrorFrame(f"Grok TTS Request Error: {e}"))

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
                    logger.warning(f"Grok TTS WebSocket closed: {msg.type}")
                    await self._handle_socket_failure(
                        f"Grok TTS WebSocket closed: {msg.type}",
                        from_receive_task=True,
                    )
                    break
        except asyncio.CancelledError:
            logger.debug("Grok TTS receive task cancelled")
        except Exception as e:
            logger.error(f"Grok TTS stream error: {e}", exc_info=True)
            await self._handle_socket_failure(
                f"Grok TTS Stream Error: {e}", from_receive_task=True
            )
            await self.push_frame(ErrorFrame(f"Grok TTS Stream Error: {e}"))

    async def _handle_text_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Grok TTS non-JSON payload: {raw[:200]}")
            return

        event_type = data.get("type")
        context_id = self._active_context_id

        if event_type == "error":
            message = data.get("message", "Unknown Grok TTS error")
            logger.error(f"Grok TTS API error: {message}")
            self._context_finished.set()
            await self.push_frame(ErrorFrame(f"Grok TTS Error: {message}"))
            return

        if context_id and context_id in self._cleared_contexts:
            logger.debug(f"Dropping stale Grok chunk for cleared context {context_id}")
            return

        if event_type == "audio.delta":
            delta_b64 = data.get("delta", "")
            if not delta_b64:
                return
            audio_bytes = base64.b64decode(delta_b64)
            if not audio_bytes:
                return

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

        if event_type == "audio.done":
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

            if context_id:
                self._cleared_contexts.discard(context_id)
                if context_id in self._started_contexts:
                    self._started_contexts.discard(context_id)
                    await self.push_frame(TTSStoppedFrame(context_id=context_id))
                if context_id == self._active_context_id:
                    self._active_context_id = None
                    self._context_finished.set()
                    await self._cancel_context_watchdog()
            else:
                self._active_context_id = None
                self._context_finished.set()
                await self._cancel_context_watchdog()
                await self.push_frame(TTSStoppedFrame())

            trace_id = data.get("trace_id")
            logger.debug(
                f"Grok TTS audio.done context={context_id} trace_id={trace_id}"
            )

    # ------------------------------------------------------------------
    # Context management (interruptions, watchdogs)
    # ------------------------------------------------------------------

    async def _clear_active_context(self):
        context_id = self._active_context_id
        if not context_id:
            return

        self._cleared_contexts.add(context_id)
        self._context_buffers.pop(context_id, None)
        self._active_context_id = None
        self._context_finished.set()
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
                f"Grok TTS context {context_id} timed out after "
                f"{self._context_timeout_secs}s"
            )
            await self._handle_socket_failure(
                f"Grok TTS context {context_id} timed out",
                from_receive_task=False,
            )
            await self.push_frame(
                ErrorFrame(f"Grok TTS context {context_id} timed out")
            )
        except asyncio.CancelledError:
            raise

    async def _handle_socket_failure(self, reason: str, *, from_receive_task: bool):
        context_id = self._active_context_id
        logger.warning(
            f"Grok TTS transport failure. reason={reason} "
            f"active_context={context_id}"
        )

        self._active_context_id = None
        self._context_finished.set()
        self._cleared_contexts.clear()
        self._context_buffers.clear()
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

    def _buffer_aligned_audio(
        self, context_id: Optional[str], audio_bytes: bytes
    ) -> bytes:
        """Ensure audio is aligned to sample boundaries (e.g. 2 bytes for PCM16)."""
        alignment = _BYTES_PER_SAMPLE.get(self._codec, 2)
        if alignment <= 1:
            # No alignment needed for 1-byte codecs (mulaw, alaw)
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
        alignment = _BYTES_PER_SAMPLE.get(self._codec, 2)
        if alignment > 1 and len(buffer) % alignment != 0:
            # Pad to alignment
            buffer.extend(b"\x00" * (alignment - len(buffer) % alignment))
        return bytes(buffer)
