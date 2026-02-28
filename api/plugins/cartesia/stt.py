import audioop
from loguru import logger
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

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
            kwargs['live_options'].sample_rate = target_sample_rate
            self._language = kwargs['live_options'].language
            self._model = kwargs['live_options'].model
            
        super().__init__(**kwargs)

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
                    
                    # Intercepts AudioRawFrame
                    new_frame = AudioRawFrame(
                        audio=resampled_audio,
                        sample_rate=self._target_sample_rate,
                        num_channels=frame.num_channels
                    )
                    return await super().process_frame(new_frame, direction)
                except Exception as e:
                    logger.error(f"Error resampling Cartesia STT frame: {e}")
                    # Skip this corrupted frame
                    return
            else:
                return await super().process_frame(frame, direction)
                
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
        live_options=CartesiaLiveOptions(
            model=user_config.stt.model,
            language=language,
            encoding="pcm_s16le",
            sample_rate=target_rate,
        )
    )
