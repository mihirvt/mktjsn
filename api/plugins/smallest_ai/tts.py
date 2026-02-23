import aiohttp
import asyncio
import base64
import json
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    EndFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService

class SmallestAITTSService(TTSService):
    """
    Text-to-Speech service using Smallest AI WebSocket API (Lightning v3.1).
    """

    class InputParams:
        def __init__(
            self,
            language: str = "en",
            speed: float = 1.0,
            max_buffer_flush_ms: int = 0,
            consistency: float = 1.0,
            enhancement: bool = False,
            similarity: float = 1.0,
        ):
            self.language = language
            self.speed = speed
            self.max_buffer_flush_ms = max_buffer_flush_ms
            self.consistency = consistency
            self.enhancement = enhancement
            self.similarity = similarity

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "lightning-v3.1",
        sample_rate: int = 24000,
        timeout: int = 60,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._sample_rate = sample_rate
        self._timeout = timeout
        self._params = params

        # WebSocket connection state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish the WebSocket connection to Smallest AI."""
        if self._ws is not None and not self._ws.closed:
            return

        url = f"wss://waves-api.smallest.ai/api/v1/{self._model}/get_speech/stream?timeout={self._timeout}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            
            logger.debug(f"Connecting to Smallest AI TTS WebSocket: {url}")
            self._ws = await self._session.ws_connect(url, headers=headers)
            
            # Start the background task to receive audio
            self._receive_task = asyncio.create_task(self._receive_audio())
        except Exception as e:
            logger.error(f"Failed to connect to Smallest AI TTS: {e}")
            await self.push_frame(ErrorFrame(f"Smallest AI TTS Connection Error: {e}"))

    async def _disconnect(self):
        """Close the WebSocket connection to Smallest AI."""
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        if isinstance(frame, StartInterruptionFrame):
            logger.debug("TTS Interruption received. Flushing...")
            # We want to clear our internal buffers and tell the WS to flush
            await self._send_request(text="", flush=True)
            await self.push_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Send text to the TTS service."""
        # This is where Pipecat hands us completed sentences from aggregate TextFrames
        if not text.strip():
            yield None
            return

        logger.debug(f"Smallest AI TTS request: '{text}'")
        await self._send_request(text=text, flush=False, continue_flag=True)
        yield None

    async def _send_request(self, text: str, flush: bool = False, continue_flag: bool = True):
        """Send a synthesis request to the WebSocket."""
        await self._connect()

        if self._ws is None or self._ws.closed:
            logger.error("Smallest AI TTS WebSocket disconnected")
            return

        payload = {
            "voice_id": self._voice_id,
            "text": text,
            "max_buffer_flush_ms": self._params.max_buffer_flush_ms,
            "continue": continue_flag,
            "flush": flush,
            "language": self._params.language,
            "sample_rate": self._sample_rate,
            "speed": self._params.speed,
            "consistency": self._params.consistency,
            "enhancement": self._params.enhancement,
            "similarity": self._params.similarity,
        }

        try:
            await self._ws.send_json(payload)
        except Exception as e:
            logger.error(f"Error sending payload to Smallest AI: {e}")
            await self.push_frame(ErrorFrame(f"Smallest AI Synthesis Error: {e}"))

    async def _receive_audio(self):
        """Background task to continuously receive audio chunks from WebSocket."""
        try:
            while self._ws and not self._ws.closed:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    response = json.loads(msg.data)
                    
                    if "error" in response:
                        error_msg = response.get("message", "Unknown error")
                        logger.error(f"Smallest AI TTS Error: {error_msg}")
                        await self.push_frame(ErrorFrame(f"Smallest AI TTS Error: {error_msg}"))
                        continue

                    status = response.get("status")
                    if status == "chunk":
                        data = response.get("data", {})
                        audio_b64 = data.get("audio")
                        if audio_b64:
                            # Decode base64 to raw PCM
                            audio_bytes = base64.b64decode(audio_b64)
                            # Push raw audio frame using Pipecat's frame standard
                            frame = TTSAudioRawFrame(
                                audio=audio_bytes,
                                sample_rate=self._sample_rate,
                                num_channels=1
                            )
                            await self.push_frame(frame)
                    
                    elif status == "complete":
                        logger.debug(f"Smallest AI TTS complete for request {response.get('request_id')}")
                        
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Smallest AI receive task error: {e}")
            await self.push_frame(ErrorFrame(f"Smallest AI TTS Stream Error: {e}"))
