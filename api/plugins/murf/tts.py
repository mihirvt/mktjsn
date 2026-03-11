"""
Murf Falcon WebSocket TTS service for Pipecat.

Documented stream-input response shapes:
- {"audio": "<base64>", "context_id": "..."}
- {"final": true, "context_id": "..."}
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
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

try:
    from pipecat.services.tts_service import TTSService
except ImportError:
    from pipecat.services.ai_services import TTSService

MURF_REGION_WS_BASES = {
    "global": "wss://global.api.murf.ai/v1/speech/stream-input",
    "us-east": "wss://us-east.api.murf.ai/v1/speech/stream-input",
    "us-west": "wss://us-west.api.murf.ai/v1/speech/stream-input",
    "in": "wss://in.api.murf.ai/v1/speech/stream-input",
    "ca": "wss://ca.api.murf.ai/v1/speech/stream-input",
    "kr": "wss://kr.api.murf.ai/v1/speech/stream-input",
    "me": "wss://me.api.murf.ai/v1/speech/stream-input",
    "jp": "wss://jp.api.murf.ai/v1/speech/stream-input",
    "au": "wss://au.api.murf.ai/v1/speech/stream-input",
    "eu-central": "wss://eu-central.api.murf.ai/v1/speech/stream-input",
    "uk": "wss://uk.api.murf.ai/v1/speech/stream-input",
    "sa-east": "wss://sa-east.api.murf.ai/v1/speech/stream-input",
}


class MurfTTSService(TTSService):
    class InputParams:
        def __init__(
            self,
            voice: str = "Matthew",
            locale: str = "en-US",
            style: str = "Conversation",
            rate: int = 0,
            pitch: int = 0,
            min_buffer_size: int = 40,
            max_buffer_delay_in_ms: int = 300,
            region: str = "us-east",
        ):
            self.voice = voice
            self.locale = locale
            self.style = style
            self.rate = rate
            self.pitch = pitch
            self.min_buffer_size = min_buffer_size
            self.max_buffer_delay_in_ms = max_buffer_delay_in_ms
            self.region = region

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "FALCON",
        sample_rate: int = 16000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._model = model
        self._sample_rate = sample_rate
        self._params = params or MurfTTSService.InputParams()

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None

        self._active_context_id: Optional[str] = None
        self._header_pending_contexts: set[str] = set()
        self._cleared_contexts: set[str] = set()
        self._context_finished = asyncio.Event()
        self._context_finished.set()
        self._advanced_settings_sent = False

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
            await self._clear_active_context()
            await self.push_frame(frame, direction)
            return
        await super().process_frame(frame, direction)

    async def run_tts(
        self, text: str, context_id: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            yield None
            return

        await self._connect()
        if self._ws is None or self._ws.closed:
            yield ErrorFrame("Murf TTS WebSocket not connected")
            return

        await self._context_finished.wait()
        turn_context_id = context_id or f"murf-{uuid.uuid4()}"
        self._context_finished.clear()
        self._active_context_id = turn_context_id
        self._header_pending_contexts.add(turn_context_id)
        self._cleared_contexts.discard(turn_context_id)

        await self._send_advanced_settings()
        await self._send_voice_config(turn_context_id)
        await self._send_text(turn_context_id, text)
        yield None

    async def _connect(self):
        if self._ws is not None and not self._ws.closed:
            return

        if self._session is None:
            self._session = aiohttp.ClientSession()

        base_url = MURF_REGION_WS_BASES.get(self._params.region, MURF_REGION_WS_BASES["us-east"])
        query = urlencode(
            {
                "api-key": self._api_key,
                "model": self._model,
                "sample_rate": str(self._sample_rate),
                "channel_type": "MONO",
                "format": "WAV",
            }
        )
        ws_url = f"{base_url}?{query}"

        try:
            logger.debug(f"Murf TTS connecting to {base_url} sample_rate={self._sample_rate}")
            self._ws = await self._session.ws_connect(ws_url)
            self._receive_task = asyncio.create_task(self._receive_audio())
        except Exception as e:
            logger.error(f"Failed to connect to Murf TTS: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Murf TTS Connection Error: {e}"))

    async def _disconnect(self):
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
        self._header_pending_contexts.clear()
        self._cleared_contexts.clear()
        self._context_finished.set()
        self._advanced_settings_sent = False

    async def _send_advanced_settings(self):
        if self._advanced_settings_sent:
            return

        payload = {
            "min_buffer_size": int(self._params.min_buffer_size),
            "max_buffer_delay_in_ms": int(self._params.max_buffer_delay_in_ms),
        }
        await self._send_json(payload)
        self._advanced_settings_sent = True

    async def _send_voice_config(self, context_id: str):
        payload = {
            "context_id": context_id,
            "voice_config": {
                "voice_id": self._params.voice,
                "locale": self._params.locale,
                "style": self._params.style,
                "rate": int(self._params.rate),
                "pitch": int(self._params.pitch),
            },
        }
        await self._send_json(payload)

    async def _send_text(self, context_id: str, text: str):
        payload = {
            "context_id": context_id,
            "text": text,
            "end": True,
        }
        await self._send_json(payload)

    async def _clear_active_context(self):
        context_id = self._active_context_id
        if not context_id:
            return

        self._cleared_contexts.add(context_id)
        self._header_pending_contexts.discard(context_id)
        payload = {"context_id": context_id, "clear": True}
        await self._send_json(payload)
        self._active_context_id = None
        self._context_finished.set()

    async def _send_json(self, payload: dict):
        await self._connect()
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Murf TTS WebSocket disconnected")

        try:
            logger.debug(f"Murf TTS payload: {json.dumps(payload)[:300]}")
            await self._ws.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send payload to Murf TTS: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Murf TTS Request Error: {e}"))

    async def _receive_audio(self):
        try:
            while self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_text_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"Murf TTS WebSocket closed: {msg.type}")
                    break
        except asyncio.CancelledError:
            logger.debug("Murf TTS receive task cancelled")
        except Exception as e:
            logger.error(f"Murf TTS stream error: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Murf TTS Stream Error: {e}"))

    async def _handle_text_message(self, payload: str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(f"Murf TTS returned non-JSON payload: {payload[:200]}")
            return

        if "error" in data:
            message = data.get("message") or data.get("error") or "Unknown Murf TTS error"
            logger.error(f"Murf TTS API error: {message}")
            self._context_finished.set()
            await self.push_frame(ErrorFrame(f"Murf TTS Error: {message}"))
            return

        context_id = data.get("context_id")
        if context_id and context_id in self._cleared_contexts:
            logger.debug(f"Dropping stale Murf chunk for cleared context {context_id}")
            return

        if "audio" in data:
            audio_bytes = base64.b64decode(data["audio"])
            if context_id and context_id in self._header_pending_contexts and len(audio_bytes) > 44:
                audio_bytes = audio_bytes[44:]
                self._header_pending_contexts.discard(context_id)

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

        if data.get("final") is True:
            if context_id:
                self._header_pending_contexts.discard(context_id)
                self._cleared_contexts.discard(context_id)
                if context_id == self._active_context_id:
                    self._active_context_id = None
                    self._context_finished.set()
            else:
                self._active_context_id = None
                self._context_finished.set()
            logger.debug(f"Murf TTS final received for context {context_id}")
