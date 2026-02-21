import base64
import json
import asyncio
import aiohttp
import time
import time
import wave
import io
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

try:
    from pipecat.frames.frames import TTSAudioRawFrame
except ImportError:
    # Fallback to OutputAudioRawFrame if TTSAudioRawFrame is missing
    from pipecat.frames.frames import OutputAudioRawFrame as TTSAudioRawFrame

class GeminiTTSService(TTSService):
    """
    Pipcat TTS Service utilizing Gemini 2.5's REST streamGenerateContent API
    to output raw synthesized audio without websockets.
    """
    def __init__(
        self,
        *,
        api_key: str,
        voice_name: str = "Zephyr",
        voice_prompt: str = "",
        model: str = "gemini-2.5-flash-preview-tts",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.api_key = api_key
        self.voice_name = voice_name
        self.voice_prompt = voice_prompt
        # Gemini expects 'models/' prefix but we will handle that in the URL
        self.model = model
        self.session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Increase internal read buffer size because Gemini audio chunk Base64 strings can easily exceed the default 64KB aiohttp line length limit
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(), read_bufsize=1048576)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self.session:
            await self.session.close()
            self.session = None

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def run_tts(self, text: str, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        context_id = args[0] if len(args) > 0 else kwargs.get("context_id")
        
        if not self.session:
            logger.error("Aiohttp session not initialized for GeminiTTSService")
            yield ErrorFrame(error="Aiohttp session not initialized")
            return

        logger.debug(f"Generating Gemini TTS for model '{self.model}' with voice/prompt: '{self.voice_name}'")
        
        # We use standard SSE streaming endpoint as required by Gemini TTS Beta
        uri = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        
        try:
            yield TTSStartedFrame()
            
            # Combine text and prompt if provided
            final_text = f"({self.voice_prompt}) {text}" if self.voice_prompt else text

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": final_text}]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self.voice_name
                            }
                        }
                    }
                }
            }
            
            
            start_time = time.time()
            async with self.session.post(uri, json=payload) as resp:
                headers_time = time.time()
                logger.debug(f"Gemini TTS: HTTP Headers received in {headers_time - start_time:.3f}s")
                
                if resp.status != 200:
                    err_txt = await resp.text()
                    logger.error(f"Gemini TTS API error {resp.status}: {err_txt}")
                    yield ErrorFrame(error=f"Gemini TTS Error {resp.status}: {err_txt}")
                    return

                # Read Server-Sent Events (SSE)
                first_chunk_logged = False
                first_audio_logged = False
                async for line in resp.content:
                    if not line:
                        continue
                        
                    if not first_chunk_logged:
                        first_chunk_logged = True
                        logger.debug(f"Gemini TTS: First SSE chunk received in {time.time() - start_time:.3f}s")
                        
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
                                    if "inlineData" in part:
                                        b64_str = part["inlineData"].get("data")
                                        if b64_str:
                                            audio_bytes = base64.b64decode(b64_str)
                                            
                                            if not first_audio_logged:
                                                first_audio_logged = True
                                                logger.debug(f"Gemini TTS: First Audio Byte decoded in {time.time() - start_time:.3f}s from start")
                                            
                                            # Gemini REST API returns WAV format (with RIFF headers).
                                            # We must strip the 44-byte WAV header cleanly to avoid playing static chunks 
                                            # and causing PyAV memory misalignment that drops downstream frames!
                                            try:
                                                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                                                    raw_pcm = wav.readframes(wav.getnframes())
                                            except Exception as e:
                                                # Fallback if it's already RAW PCM
                                                logger.warning(f"Could not parse WAV header, assuming raw PCM: {e}")
                                                raw_pcm = audio_bytes

                                            frame_kwargs = {
                                                "audio": raw_pcm,
                                                "sample_rate": self.sample_rate,
                                                "num_channels": 1
                                            }
                                            if context_id is not None:
                                                frame_kwargs["context_id"] = context_id
                                            
                                            # Yield the unified PCM to the Pipecat Audio buffer, letting internal pyAV handle timing securely
                                            yield TTSAudioRawFrame(**frame_kwargs)
                        except Exception as e:
                            logger.error(f"Failed to parse Gemini SSE chunk: {e}")
                            
        except Exception as e:
            logger.exception(f"Exception in Gemini TTS API: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()
