import audioop
from loguru import logger
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
import urllib.parse
from websockets.protocol import State
from websockets.asyncio.client import connect as websocket_connect
class CustomCartesiaSTTService(CartesiaSTTService):
    """
    Cartesia STT performs poorly with PCM 8000Hz (telephony).
    This plugin ensures we send PCM at >=16000Hz to Cartesia.
    If the pipeline uses 8000Hz, this forces an upsample to 16kHz
    before flushing the audio over WebSocket.
    """
    def __init__(self, target_sample_rate=16000, original_sample_rate=8000, **kwargs):
        self._target_sample_rate = target_sample_rate
        self._original_sample_rate = original_sample_rate
        self._resample_state = None
        
        # Override sample rate to what we actually send to Cartesia socket
        kwargs['sample_rate'] = target_sample_rate
        
        if 'live_options' in kwargs:
            # Older Pipecat versions pass `CartesiaLiveOptions` directly containing attributes
            self._language = getattr(kwargs['live_options'], 'language', 'hi')
            self._model = getattr(kwargs['live_options'], 'model', 'ink-whisper')
            # CartesiaLiveOptions also defaults encoding
            self._encoding = getattr(kwargs['live_options'], 'encoding', 'pcm_s16le')
            kwargs['live_options'].sample_rate = target_sample_rate
            
            self._min_volume = kwargs.pop('min_volume', 0.1)
            self._max_silence_duration_secs = kwargs.pop('max_silence_duration_secs', 1.5)
            
        super().__init__(**kwargs)
        
        # We stream strictly in real-time, matching other STT implementations.
        pass

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug(f"Connecting to Cartesia STT Plugin with Strict VAD (Vol={self._min_volume}, Silence={self._max_silence_duration_secs})")

            params = {
                "model": self._model,
                "encoding": self._encoding,
                "sample_rate": str(self.sample_rate),
                "min_volume": str(self._min_volume),
                "max_silence_duration_secs": str(self._max_silence_duration_secs)
            }
            if self._language and self._language not in ["multilingual", "multi"]:
                params["language"] = self._language
            ws_url = f"wss://{self._base_url}/stt/websocket?{urllib.parse.urlencode(params)}"
            headers = {"Cartesia-Version": "2025-04-16", "X-API-Key": self._api_key}

            self._websocket = await websocket_connect(ws_url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            if frame.sample_rate != self._target_sample_rate:
                try:
                    # audioop.ratecv(fragment, width, nchannels, inrate, outrate, weightA)
                    resampled_audio, self._resample_state = audioop.ratecv(
                        frame.audio,
                        2, # 16-bit PCM = 2 bytes
                        frame.num_channels,
                        frame.sample_rate,
                        self._target_sample_rate,
                        self._resample_state
                    )
                    
                    import copy
                    new_frame = copy.copy(frame)
                    new_frame.audio = bytes(resampled_audio)
                    new_frame.sample_rate = self._target_sample_rate
                    if hasattr(new_frame, "num_frames"):
                        new_frame.num_frames = int(len(new_frame.audio) / (new_frame.num_channels * 2))
                        
                    await super().process_frame(new_frame, direction)
                    return
                except Exception as e:
                    logger.error(f"Error resampling Cartesia STT frame: {e}")
                    # Skip this corrupted frame
                    return
            else:
                await super().process_frame(frame, direction)
                return
                
        # Forward any non-audio frames untouched
        return await super().process_frame(frame, direction)

def create_cartesia_stt(user_config, audio_config):
    language = getattr(user_config.stt, "language", "hi") or "hi"
    
    transport_rate = audio_config.transport_in_sample_rate
    
    # Cartesia STT explicitly supports only: 8000 (mulaw), 16000, 24000, 44100.
    # WebRTC standard is 48000Hz. If we pass 48000 to Cartesia, it hallucinates wildly.
    # Telephony provides 8000Hz, which also degrades PCM STT.
    # Therefore, we universally force-resample ALL inputs to 16000Hz.
    target_rate = 16000
    
    logger.info(
        f"[Cartesia STT Plugin] Initialized: transport={transport_rate}Hz -> cartesia={target_rate}Hz, lang={language}"
    )
    
    return CustomCartesiaSTTService(
        api_key=user_config.stt.api_key,
        target_sample_rate=target_rate,
        original_sample_rate=transport_rate,
        min_volume=0.1,  # Threshold to reject background noise/TTS echo
        max_silence_duration_secs=1.5,  # Prevents long open chunks collecting noise
        live_options=CartesiaLiveOptions(
            model=user_config.stt.model,
            language=language,
            encoding="pcm_s16le",
            sample_rate=target_rate,
        )
    )
