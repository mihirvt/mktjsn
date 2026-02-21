import base64
import json
import asyncio
import aiohttp
from typing import AsyncGenerator
from loguru import logger

from pipecat.services.ai_services import TTSService
from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
)

class GeminiTTSService(TTSService):
    """
    Pipcat TTS Service utilizing Gemini 2.5's native BidiGenerateContent
    full duplex websocket for high quality, low latency audio streaming.
    """
    def __init__(
        self,
        *,
        api_key: str,
        voice_prompt: str,
        model: str = "gemini-2.5-flash-tts-lite",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.voice_prompt = voice_prompt
        self.model = model
        self.sample_rate = sample_rate
        self.session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self.session = aiohttp.ClientSession()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self.session:
            await self.session.close()
            self.session = None

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not self.session:
            logger.error("Aiohttp session not initialized for GeminiTTSService")
            yield ErrorFrame(error="Aiohttp session not initialized")
            return

        logger.debug(f"Generating Gemini TTS for model '{self.model}' with prompt: '{self.voice_prompt}'")
        uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        
        try:
            yield TTSStartedFrame()
            
            async with self.session.ws_connect(uri) as ws:
                setup_msg = {
                    "setup": {
                        "model": f"models/{self.model}",
                        "generationConfig": {
                            "responseModalities": ["AUDIO"]
                        },
                        "systemInstruction": {
                            "parts": [{"text": self.voice_prompt}]
                        }
                    }
                }
                await ws.send_json(setup_msg)
                
                # Wait for setup response
                setup_resp = await ws.receive_json()
                logger.debug(f"Gemini Setup Response: {setup_resp}")
                
                # Send text for synthesis
                client_content = {
                    "clientContent": {
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": text}]
                            }
                        ],
                        "turnComplete": True
                    }
                }
                await ws.send_json(client_content)
                
                while True:
                    msg = await ws.receive_json()
                    
                    if "serverContent" in msg:
                        content = msg["serverContent"]
                        if "modelTurn" in content:
                            parts = content["modelTurn"].get("parts", [])
                            for part in parts:
                                if "inlineData" in part:
                                    b64_audio = part["inlineData"].get("data")
                                    if b64_audio:
                                        audio_bytes = base64.b64decode(b64_audio)
                                        yield AudioRawFrame(
                                            audio=audio_bytes,
                                            sample_rate=self.sample_rate,
                                            num_channels=1
                                        )
                        if content.get("turnComplete"):
                            break
                    elif "error" in msg:
                        error_msg = msg["error"]
                        logger.error(f"Gemini TTS Error: {error_msg}")
                        yield ErrorFrame(error=str(error_msg))
                        break
                        
        except Exception as e:
            logger.exception(f"Exception in Gemini TTS streaming: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()
