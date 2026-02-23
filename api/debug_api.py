import asyncio
from api.routes.user import get_default_configurations
import time

async def main():
    print("Testing get_default_configurations...")
    start = time.time()
    res = await get_default_configurations()
    print("Done in", time.time() - start)
    print("TTS providers:", list(res["tts"].keys()))

try:
    asyncio.run(main())
except Exception as e:
    print("Error:", e)
