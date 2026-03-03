import os
import asyncio
import json
import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException
from groq import RateLimitError, APIStatusError

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# --- MODIFIED: Usage file stores data per-model ---
USAGE_FILE = 'groq_usage.json'

class GroqAPI:
    """
    A simple, asynchronous client to query the Groq API
    using langchain-groq, with persistent *daily* usage tracking
    *per model* and rate limit awareness.
    """
    
    def __init__(self, model_id: str = "llama3-8b-8192", temperature: float = 0.7):
        """
        Initializes the client with a specific model.
        """
        
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise EnvironmentError(
                "No API key found. Please set 'GROQ_API_KEY' in your .env file."
            )
        
        try:
            self.chat_model = ChatGroq(
                temperature=temperature,
                model_name=model_id,
                api_key=api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize ChatGroq for model '{model_id}'. Error: {e}")

        self.model_id = model_id
        
        # --- MODIFIED: Usage tracking properties ---
        self.last_response_metadata = None
        # --- NEW: Store last seen rate limit info ---
        self.last_rate_limit_info = {}
        # --- MODIFIED: Tracks daily tokens for *this* model ---
        self.model_daily_tokens = 0 
        self.last_usage_date = ""  
        
        self.usage_lock = asyncio.Lock()
        
        # Load initial usage for this model
        self._load_daily_usage_sync()


    def _load_daily_usage_sync(self):
        """
        Synchronous version of load_daily_usage for use in __init__.
        Loads the daily usage for *this specific model*.
        """
        today_str = datetime.date.today().isoformat()
        
        try:
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # File missing or corrupt, create a new one
            data = {}

        saved_date = data.get('date')
        
        if saved_date == today_str:
            # File is from today, load this model's data
            model_data = data.get('models', {}).get(self.model_id, {})
            self.model_daily_tokens = model_data.get('total_tokens_used', 0)
        else:
            # It's a new day, reset
            self.model_daily_tokens = 0
            # We'll save a fresh file on the first query
        
        self.last_usage_date = today_str


    def _save_daily_usage_sync(self):
        """
        Synchronous version of save_daily_usage for use in __init__.
        Saves the daily usage for *all* models.
        """
        today_str = datetime.date.today().isoformat()
        try:
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
            if data.get('date') != today_str:
                # New day, clear old data
                data = {'date': today_str, 'models': {}}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'date': today_str, 'models': {}}

        # Ensure model key exists
        if self.model_id not in data.get('models', {}):
            data.setdefault('models', {})[self.model_id] = {}
        
        # Update tokens for this model
        data['models'][self.model_id]['total_tokens_used'] = self.model_daily_tokens
        data['date'] = today_str

        with open(USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    async def _load_daily_usage(self):
        """
        Asynchronously loads the current usage data from the file.
        Resets the counter if it's a new day.
        """
        today_str = datetime.date.today().isoformat()
        try:
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        saved_date = data.get('date')
        
        if saved_date == today_str:
            model_data = data.get('models', {}).get(self.model_id, {})
            self.model_daily_tokens = model_data.get('total_tokens_used', 0)
        else:
            self.model_daily_tokens = 0
        
        self.last_usage_date = today_str


    async def _save_daily_usage(self):
        """
        Asynchronously saves the current usage data to the file.
        """
        today_str = self.last_usage_date
        
        # --- MODIFIED: Read-Modify-Write to safely update ---
        try:
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
            if data.get('date') != today_str:
                # New day, clear old data
                data = {'date': today_str, 'models': {}}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'date': today_str, 'models': {}}

        # Ensure model key exists
        if self.model_id not in data.get('models', {}):
            data.setdefault('models', {})[self.model_id] = {}
        
        # Update tokens for this model
        data['models'][self.model_id]['total_tokens_used'] = self.model_daily_tokens
        data['date'] = today_str

        with open(USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
    async def get_current_model_daily_total(self) -> int:
        """
        Public method to safely get the current daily total for this model.
        """
        async with self.usage_lock:
            await self._load_daily_usage()
            return self.model_daily_tokens


    async def query(self, prompt: str) -> str:
        """
        Queries the Groq model, now with persistent daily usage tracking.
        """
        messages = [HumanMessage(content=prompt)]
        
        try:
            response_message = await self.chat_model.ainvoke(messages)
            
            if response_message.response_metadata:
                self.last_response_metadata = response_message.response_metadata
                
                # --- NEW: Store Rate Limit Info ---
                self.last_rate_limit_info = {
                    "limit_requests": self.last_response_metadata.get('x_ratelimit_limit_requests'),
                    "remaining_requests": self.last_response_metadata.get('x_ratelimit_remaining_requests'),
                    "limit_tokens": self.last_response_metadata.get('x_ratelimit_limit_tokens'),
                    "remaining_tokens": self.last_response_metadata.get('x_ratelimit_remaining_tokens'),
                    "reset_requests": self.last_response_metadata.get('x_ratelimit_reset_requests'),
                    "reset_tokens": self.last_response_metadata.get('x_ratelimit_reset_tokens'),
                }
                
                # --- MODIFIED: Usage Tracking ---
                token_usage = self.last_response_metadata.get('token_usage', {})
                tokens_this_call = token_usage.get('total_tokens', 0)
                
                async with self.usage_lock:
                    # 1. Load the most recent total from file
                    await self._load_daily_usage()
                    # 2. Add this call's tokens
                    self.model_daily_tokens += tokens_this_call
                    # 3. Save the new total back to file
                    await self._save_daily_usage()
            # --- End Usage Tracking ---

            return response_message.content

        except RateLimitError as e:
            return f"Groq API Error: Rate limit exceeded. {e}"
        except APIStatusError as e:
            if e.status_code == 401:
                return f"Groq API Error: 401 Unauthorized. Check your GROQ_API_KEY. {e}"
            if e.status_code == 400:
                 return f"Groq API Error: 400 Bad Request. Check your model_id ('{self.model_id}') or prompt. {e}"
            return f"Groq API Error: An API error occurred (Status {e.status_code}). {e}"
        except OutputParserException as e:
            return f"Groq API Error: Failed to parse model output. {e}"
        except Exception as e:
            return f"Groq API Error: An unexpected error occurred. {e}"

# --- Example Usage (MODIFIED) ---
async def main():
    """
    A simple example of how to use the GroqAPI class.
    """
    try:
        # --- Client 1 (Llama 3 8B) ---
        client_llama = GroqAPI(model_id="llama3-8b-8192", temperature=0.1)
        
        prompt1 = "Hello! What's the capital of France?"
        print(f"--- Querying {client_llama.model_id}: '{prompt1}' ---")
        response1 = await client_llama.query(prompt1)
        print(f"Response: {response1}\n")
        
        print(f"Total Daily Tokens ({client_llama.model_id}): {await client_llama.get_current_model_daily_total()}")
        if client_llama.last_rate_limit_info:
            print(f"Per-Minute Limit (Tokens): "
                  f"{client_llama.last_rate_limit_info.get('remaining_tokens')} / "
                  f"{client_llama.last_rate_limit_info.get('limit_tokens')}")

        # --- Client 2 (Mixtral) ---
        client_mixtral = GroqAPI(model_id="mixtral-8x7b-32768", temperature=0.1)
        
        prompt2 = "What is the primary ingredient in guacamole?"
        print(f"\n--- Querying {client_mixtral.model_id}: '{prompt2}' ---")
        response2 = await client_mixtral.query(prompt2)
        print(f"Response: {response2}\n")
        
        print(f"Total Daily Tokens ({client_mixtral.model_id}): {await client_mixtral.get_current_model_daily_total()}")
        if client_mixtral.last_rate_limit_info:
            print(f"Per-Minute Limit (Tokens): "
                  f"{client_mixtral.last_rate_limit_info.get('remaining_tokens')} / "
                  f"{client_mixtral.last_rate_limit_info.get('limit_tokens')}")
        
        # --- Re-Query Client 1 ---
        prompt3 = "And what's its main river?"
        print(f"\n--- Querying {client_llama.model_id}: '{prompt3}' ---")
        response3 = await client_llama.query(prompt3)
        print(f"Response: {response3}\n")

        print(f"Total Daily Tokens ({client_llama.model_id}): {await client_llama.get_current_model_daily_total()}")
        if client_llama.last_rate_limit_info:
            print(f"Per-Minute Limit (Tokens): "
                  f"{client_llama.last_rate_limit_info.get('remaining_tokens')} / "
                  f"{client_llama.last_rate_limit_info.get('limit_tokens')}")


    except (EnvironmentError, ValueError) as e:
        print(f"Failed to start client: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())