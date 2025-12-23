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
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://github.com/santos-sanz/LLM_Vending_Machine",
                        "X-Title": "LLM Vending Machine Simulation",
                    }
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10 # Increase wait time
                    print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"Error calling OpenRouter: {e}")
                    return None
        return None
