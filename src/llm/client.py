import os
from openai import OpenAI
from src.config import OPENROUTER_API_KEY, DEFAULT_MODEL

class OpenRouterClient:
    def __init__(self, api_key=None, base_url="https://openrouter.ai/api/v1"):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    def complete(self, prompt, model=None, system_prompt="You are a helpful assistant.", max_retries=3):
        import time
        if not self.client:
            print("Error: OpenAI client not initialized. Check your API key.")
            return None
            
        model = model or DEFAULT_MODEL
        # Log which model is being used for debugging purposes
        print(f"[LLM Client] Calling OpenRouter with model: {model}")
        for attempt in range(max_retries):
            try:
                # Adding a timeout to prevent hanging indefinitely
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://github.com/santos-sanz/LLM_Vending_Machine",
                        "X-Title": "LLM Vending Machine Simulation",
                    },
                    timeout=30.0  # 30 seconds timeout
                )
                
                if not response.choices[0].message.content:
                    print(f"[LLM Client] Warning: Received empty response from model {model}.")
                    return None
                    
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"[LLM Client] Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                elif "timeout" in error_str.lower():
                    print(f"[LLM Client] Timeout error on attempt {attempt + 1}: {error_str}")
                    if attempt < max_retries - 1:
                        time.sleep(2) # Brief pause before retry
                    else:
                        return None
                else:
                    print(f"[LLM Client] Error calling OpenRouter: {error_str}")
                    return None
        return None


class OllamaClient:
    """Client for local model inference using Ollama."""
    def __init__(self, base_url=None):
        from src.config import OLLAMA_BASE_URL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",  # Ollama doesn't require a real API key
        )

    def complete(self, prompt, model=None, system_prompt="You are a helpful assistant.", max_retries=3):
        import time
        from src.config import DEFAULT_LOCAL_MODEL
        
        model = model or DEFAULT_LOCAL_MODEL
        print(f"[LLM Client] Calling Ollama with model: {model}")
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=60.0  # Longer timeout for local models
                )
                
                if not response.choices[0].message.content:
                    print(f"[LLM Client] Warning: Received empty response from model {model}.")
                    return None
                    
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                print(f"[LLM Client] Error calling Ollama on attempt {attempt + 1}: {error_str}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
        return None


def create_llm_client(mode="online", model_name=None):
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        mode: "online" for OpenRouter, "local" for Ollama
        model_name: Optional model name override
        
    Returns:
        LLM client instance
    """
    if mode == "local":
        return OllamaClient()
    elif mode == "online":
        return OpenRouterClient()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'online'.")
