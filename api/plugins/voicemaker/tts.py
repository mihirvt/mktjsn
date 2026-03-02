"""
Voicemaker TTS WebSocket Service for Pipecat.

Official API reference payload (do NOT rename fields):
{
    "VoiceId": "ai3-Jony",
    "Text": "Welcome to Voicemaker API.",
    "LanguageCode": "en-US",
    "OutputFormat": "mp3",
    "SampleRate": "48000",
    "MasterVolume": "0",
    "MasterSpeed": "0",
    "MasterPitch": "0"
}

WebSocket endpoint: wss://developer.voicemaker.in/api/v1/voice/convert
Auth: Authorization: Bearer <API_KEY>

Response chunks: { "success": true, "audio": "<base64>" }
Final chunk:     { "success": true, "audio": "<base64>", "isFinal": true }
Error:           { "success": false, "message": "...", "errors": [...] }

NOTE: The API streams base64-encoded MP3 chunks per message. Pipecat expects
raw PCM bytes (TTSAudioRawFrame). We decode base64 → MP3 bytes, then use
pydub to decode MP3 → 16-bit PCM at the requested sample rate.
"""

import asyncio
import base64
import io
import json
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

try:
    from pipecat.services.tts_service import TTSService
except ImportError:
    from pipecat.services.ai_services import TTSService

VOICEMAKER_WS_URL = "wss://developer.voicemaker.in/api/v1/voice/convert"


def _decode_mp3_to_pcm(mp3_bytes: bytes, sample_rate: int) -> bytes:
    """Decode MP3 bytes to raw 16-bit PCM at the given sample rate."""
    try:
        from pydub import AudioSegment  # type: ignore

        segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        segment = segment.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        return segment.raw_data
    except Exception as e:
        logger.error(f"Voicemaker: failed to decode MP3 to PCM: {e}", exc_info=True)
        return b""


class VoicemakerTTSService(TTSService):
    """
    Text-to-Speech service using the Voicemaker WebSocket API.

    Streams base64-encoded MP3 audio chunks and converts them to raw 16-bit
    PCM for Pipecat's audio pipeline. One persistent WebSocket connection is
    maintained for the lifetime of the pipeline; each run_tts call sends a
    new JSON payload and receives the corresponding audio back.
    """

    # ------------------------------------------------------------------
    # InputParams – mirrors the Voicemaker API optional fields

    class InputParams:
        def __init__(
            self,
            language_code: str = "en-US",
            output_format: str = "mp3",
            master_volume: str = "0",
            master_speed: str = "0",
            master_pitch: str = "0",
            # Pro/ProPlus-only
            stability: Optional[str] = None,
            similarity: Optional[str] = None,
            pro_engine: Optional[str] = None,
            # Optional
            accent_code: Optional[str] = None,
        ):
            self.language_code = language_code
            self.output_format = output_format
            self.master_volume = master_volume
            self.master_speed = master_speed
            self.master_pitch = master_pitch
            self.stability = stability
            self.similarity = similarity
            self.pro_engine = pro_engine
            self.accent_code = accent_code

    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "ai3-Jony",
        sample_rate: int = 16000,
        params: Optional["VoicemakerTTSService.InputParams"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._params = params or VoicemakerTTSService.InputParams()

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._current_context_id: Optional[str] = None

        # Per-request: accumulate raw MP3 bytes until isFinal
        self._mp3_buffer: bytes = b""

    # ------------------------------------------------------------------
    # Lifecycle

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    # ------------------------------------------------------------------
    # Connect / Disconnect

    async def _connect(self):
        if self._ws is not None and not self._ws.closed:
            return

        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            logger.debug(f"Voicemaker TTS: connecting to {VOICEMAKER_WS_URL}")
            self._ws = await self._session.ws_connect(VOICEMAKER_WS_URL, headers=headers)
            logger.info("Voicemaker TTS: WebSocket connected")
            self._receive_task = asyncio.create_task(self._receive_audio())
        except Exception as e:
            logger.error(f"Voicemaker TTS: connection failed: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Voicemaker TTS Connection Error: {e}"))

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

    # ------------------------------------------------------------------
    # TTS

    async def run_tts(
        self, text: str, context_id: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            yield None
            return

        self._current_context_id = context_id
        self._mp3_buffer = b""  # reset buffer for new request

        logger.debug(f"Voicemaker TTS request: '{text[:80]}...' context_id={context_id}")
        await self._send_request(text)
        yield None  # audio is pushed from _receive_audio background task

    async def _send_request(self, text: str):
        await self._connect()

        if self._ws is None or self._ws.closed:
            logger.error("Voicemaker TTS: WebSocket not connected")
            return

        p = self._params
        payload: dict = {
            "VoiceId": self._voice_id,
            "Text": text,
            "LanguageCode": p.language_code,
            "OutputFormat": p.output_format,
            # SampleRate for Voicemaker is a STRING per API spec
            "SampleRate": str(self._sample_rate),
            "MasterVolume": str(p.master_volume),
            "MasterSpeed": str(p.master_speed),
            "MasterPitch": str(p.master_pitch),
        }

        # Optional fields – only send if provided
        if p.stability is not None:
            payload["Stability"] = str(p.stability)
        if p.similarity is not None:
            payload["Similarity"] = str(p.similarity)
        if p.pro_engine is not None:
            payload["ProEngine"] = p.pro_engine
        if p.accent_code is not None:
            payload["AccentCode"] = p.accent_code

        try:
            logger.debug(f"Voicemaker sending payload: {json.dumps(payload)[:300]}")
            await self._ws.send_json(payload)
        except Exception as e:
            logger.error(f"Voicemaker TTS: send error: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Voicemaker TTS send error: {e}"))

    # ------------------------------------------------------------------
    # Receive loop

    async def _receive_audio(self):
        """Background task to receive audio chunks from the WebSocket."""
        logger.debug("Voicemaker TTS: receive task started")
        try:
            while self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning(f"Voicemaker TTS: non-JSON message: {msg.data[:200]}")
                        continue

                    if not data.get("success", True):
                        errors = data.get("errors", [])
                        err_msg = data.get("message", "Unknown error")
                        logger.error(f"Voicemaker TTS error: {err_msg} – {errors}")
                        await self.push_frame(ErrorFrame(f"Voicemaker TTS Error: {err_msg}"))
                        continue

                    audio_b64 = data.get("audio", "")
                    is_final = data.get("isFinal", False)

                    if audio_b64:
                        chunk_bytes = base64.b64decode(audio_b64)
                        self._mp3_buffer += chunk_bytes
                        logger.debug(
                            f"Voicemaker: received audio chunk "
                            f"{len(chunk_bytes)} bytes (isFinal={is_final})"
                        )

                    if is_final and self._mp3_buffer:
                        # Whole MP3 accumulated — now decode to PCM and push
                        pcm_bytes = _decode_mp3_to_pcm(self._mp3_buffer, self._sample_rate)
                        self._mp3_buffer = b""

                        if pcm_bytes:
                            frame_kwargs = {
                                "audio": pcm_bytes,
                                "sample_rate": self._sample_rate,
                                "num_channels": 1,
                            }
                            if self._current_context_id:
                                frame_kwargs["context_id"] = self._current_context_id
                            try:
                                frame = TTSAudioRawFrame(**frame_kwargs)
                            except TypeError:
                                frame = TTSAudioRawFrame(
                                    audio=pcm_bytes,
                                    sample_rate=self._sample_rate,
                                    num_channels=1,
                                )
                            await self.push_frame(frame)
                            logger.debug(
                                f"Voicemaker: pushed {len(pcm_bytes)} PCM bytes "
                                f"@ {self._sample_rate}Hz"
                            )

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"Voicemaker TTS: WS closed/error (type={msg.type})")
                    break
                else:
                    logger.debug(f"Voicemaker TTS: unhandled WS message type: {msg.type}")

        except asyncio.CancelledError:
            logger.debug("Voicemaker TTS: receive task cancelled")
        except Exception as e:
            logger.error(f"Voicemaker TTS: receive task error: {e}", exc_info=True)
            await self.push_frame(ErrorFrame(f"Voicemaker TTS Stream Error: {e}"))
