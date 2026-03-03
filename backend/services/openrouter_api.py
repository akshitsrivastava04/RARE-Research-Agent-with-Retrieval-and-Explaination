import aiohttp
import asyncio

class OpenRouterAPI:
    @staticmethod
    async def query(api_key: str, model: str, prompt: str):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                try:
                    data = await resp.json()
                except Exception as e:
                    return f"Error parsing JSON: {e}"

                # Handle errors from the API
                if 'error' in data:
                    return f"OpenRouter API Error: {data['error']}"
                
                if 'choices' not in data or len(data['choices']) == 0:
                    return f"Unexpected API response: {data}"

                return data['choices'][0]['message']['content']
