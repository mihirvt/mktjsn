import asyncio
import aiohttp
import json

async def test():
    api_key = "05c72410-1614-11f1-9acc-b9e090d5fe67"
    ws_url = f"wss://developer.voicemaker.in/api/v1/voice/convert"
    
    methods = [
        {"name": "Standard Header", "headers": {"Authorization": f"Bearer {api_key}"}, "url": ws_url, "payload": {"VoiceId": "proplus-Priya", "Text": "Testing 123.", "LanguageCode": "en-US", "OutputFormat": "mp3", "SampleRate": "48000"}},
        {"name": "Upper Case Header", "headers": {"AUTHORIZATION": f"Bearer {api_key}"}, "url": ws_url, "payload": {"VoiceId": "proplus-Priya", "Text": "Testing 123.", "LanguageCode": "en-US", "OutputFormat": "mp3", "SampleRate": "48000"}},
        {"name": "No header, token in query", "headers": {}, "url": f"{ws_url}?token={api_key}", "payload": {"VoiceId": "proplus-Priya", "Text": "Testing 123.", "LanguageCode": "en-US", "OutputFormat": "mp3", "SampleRate": "48000"}},
        {"name": "API key direct in dict without Bearer", "headers": {"Authorization": api_key}, "url": ws_url, "payload": {"VoiceId": "proplus-Priya", "Text": "Testing 123.", "LanguageCode": "en-US", "OutputFormat": "mp3", "SampleRate": "48000"}},
        {"name": "Payload Authorization without Bearer", "headers": {"Authorization": f"Bearer {api_key}"}, "url": ws_url, "payload": {"Authorization": api_key, "VoiceId": "proplus-Priya", "Text": "Testing 123.", "LanguageCode": "en-US", "OutputFormat": "mp3", "SampleRate": "48000"}},
    ]
    
    async with aiohttp.ClientSession() as session:
        for m in methods:
            print(f"Testing {m['name']}...")
            try:
                async with session.ws_connect(m['url'], headers=m['headers']) as ws:
                    print("  WS Connected!")
                    await ws.send_json(m['payload'])
                    print("  JSON sent.")
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    print("  Received:", msg.data[:200] if msg.type == aiohttp.WSMsgType.TEXT else f"Type: {msg.type}")
            except Exception as e:
                print("  Exception:", e)

if __name__ == "__main__":
    asyncio.run(test())
