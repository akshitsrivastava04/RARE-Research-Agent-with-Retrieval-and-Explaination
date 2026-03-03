import os
import asyncio
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException

# Load environment variables (e.g., HF_API_KEY)
load_dotenv()

class HuggingFaceAPI:
    """
    A simple, asynchronous client to query the Hugging Face Inference API
    using langchain-huggingface.
    
    This class is instantiated for a *specific model*.
    """
    
    def __init__(self, model_id: str, temperature: float = 0.7):
        """
        Initializes the client with a specific model.

        :param model_id: The ID of the model to query (e.g., 'mistralai/Mistral-7B-Instruct-v0.1').
        :param temperature: The generation temperature.
        """
        
        # ✅ Get the API key from your preferred env variable
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            # Fallback to the langchain default
            api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not api_key:
            raise EnvironmentError(
                "No API key found. Please set 'HF_API_KEY' in your .env file."
            )
        
        # 1. Set up the low-level endpoint
        try:
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model_id,
                task="conversational",
                temperature=temperature,
                huggingfacehub_api_token=api_key,
                max_new_tokens=512 
            )
        except Exception as e:
            # This helps catch bad model IDs early
            raise ValueError(f"Failed to initialize HuggingFaceEndpoint for model '{model_id}'. Error: {e}")

        # 2. Wrap it in the ChatHuggingFace interface
        self.chat_model = ChatHuggingFace(llm=llm_endpoint)
        self.model_id = model_id


    async def query(self, prompt: str) -> str:
        """
        Queries the Hugging Face model with a single prompt.

        :param prompt: The input text prompt for the model.
        :return: The generated text response or an error message string.
        """
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            # Use the asynchronous 'ainvoke' method
            response_message = await self.chat_model.ainvoke(messages)
            return response_message.content

        except OutputParserException as e:
            return f"Hugging Face API Error: Failed to parse model output. {e}"
        except Exception as e:
            # LangChain raises exceptions for API errors (503, 401, 403, etc.)
            error_str = str(e)
            if "503" in error_str and "is loading" in error_str:
                return (
                    f"Hugging Face API Error: Model '{self.model_id}' is loading. "
                    f"Please wait a few moments and try again."
                )
            if "403" in error_str:
                return (
                    f"Hugging Face API Error: 403 Forbidden. You may not have "
                    f"permission for model '{self.model_id}'. Error: {e}"
                )
            return f"Hugging Face API Error: An unexpected error occurred. {e}"