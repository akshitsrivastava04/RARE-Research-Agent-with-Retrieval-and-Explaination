import httpx
import os
import asyncio
from dotenv import load_dotenv  


load_dotenv()

# We'll use the environment variable for the API key,
# but as per instructions, we'll leave it blank in the code
# as the Canvas environment will provide it.
API_KEY = os.getenv("GOOGLE_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

SUMMARIZER_SYSTEM_PROMPT = (
    "You are a highly skilled text summarizer. Your sole purpose is to "
    "summarize the following context into a dense, single paragraph. "
    "Focus *only* on the key facts, entities, and information relevant to "
    "answering a potential user question. "
    "Do not add any preamble, conversational text, or "
    "markdown formatting. Output *only* the summary."
)

# Use a single, persistent async client
client = httpx.AsyncClient()

async def summarize_context(context: str) -> str:
    """
    Summarizes the given RAG context using the Gemini API.
    """
    if not context.strip():
        return "" # No context to summarize

    payload = {
        "contents": [{
            "parts": [{"text": context}]
        }],
        "systemInstruction": {
            "parts": [{"text": SUMMARIZER_SYSTEM_PROMPT}]
        },
    }

    try:
        # --- Exponential Backoff ---
        for attempt in range(3): # Retry up to 3 times
            try:
                response = await client.post(
                    API_URL,
                    headers={'Content-Type': 'application/json'},
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status() # Raise an exception for 4xx/5xx
                
                # --- Process successful response ---
                result = response.json()
                candidate = result.get("candidates", [])[0]
                if candidate and candidate.get("content", {}).get("parts", [])[0].get("text"):
                    return candidate["content"]["parts"][0]["text"].strip()
                else:
                    print("[Summarizer Error] Unexpected response structure:", result)
                    return context # Fallback to original context

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 500, 503] and attempt < 2:
                    wait_time = (2 ** attempt) # 1s, 2s
                    print(f"[Summarizer Warning] Retrying after {wait_time}s due to {e.response.status_code}")
                    await asyncio.sleep(wait_time)
                else:
                    raise # Re-raise final error
            except httpx.RequestError as e:
                # Handle network-level errors
                if attempt < 2:
                    wait_time = (2 ** attempt)
                    print(f"[Summarizer Warning] Retrying after {wait_time}s due to network error: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        print("[Summarizer Error] All retries failed.")
        return context # Fallback to original context

    except Exception as e:
        print(f"[Summarizer Error] Failed to summarize context: {e}")
        # Fallback to returning the original (unsummarized) context
        return context