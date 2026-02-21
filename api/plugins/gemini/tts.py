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
    Pipcat TTS Service utilizing Gemini 2.5's REST streamGenerateContent API
    to output raw synthesized audio without websockets.
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
        # Gemini expects 'models/' prefix but we will handle that in the URL
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
        
        # We use standard SSE streaming endpoint
        uri = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        
        try:
            yield TTSStartedFrame()
            
            payload = {
                "contents": [
                    {
                        "parts": [{"text": text}]
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": self.voice_prompt}]
                },
                "generationConfig": {
                    "responseModalities": ["AUDIO"]
                }
            }
            
            async with self.session.post(uri, json=payload) as resp:
                if resp.status != 200:
                    err_txt = await resp.text()
                    logger.error(f"Gemini TTS API error {resp.status}: {err_txt}")
                    yield ErrorFrame(error=f"Gemini TTS Error {resp.status}: {err_txt}")
                    return

                # Read Server-Sent Events (SSE)
                async for line in resp.content:
                    if not line:
                        continue
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        json_str = line_str[6:].strip()
                        if json_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(json_str)
                            if "candidates" in data and len(data["candidates"]) > 0:
                                parts = data["candidates"][0].get("content", {}).get("parts", [])
                                for part in parts:
                                    # Depending on exactly how Gemini 2.5 returns audio chunks
                                    if "inlineData" in part:
                                        b64_str = part["inlineData"].get("data")
                                        if b64_str:
                                            audio_bytes = base64.b64decode(b64_str)
                                            yield AudioRawFrame(
                                                audio=audio_bytes,
                                                sample_rate=self.sample_rate,
                                                num_channels=1
                                            )
                                    elif "text" in part:
                                        # Handle the scenario where model occasionally falls back to text modality
                                        pass
                        except Exception as e:
                            logger.error(f"Failed to parse Gemini SSE chunk: {e}")
                            
        except Exception as e:
            logger.exception(f"Exception in Gemini TTS streaming: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()
