from src.llm.client import OpenRouterClient

def test_connectivity():
    print("Testing OpenRouter connectivity...")
    client = OpenRouterClient()
    
    if not client.api_key:
        print("Error: OPENROUTER_API_KEY not found in .env or environment.")
        return

    prompt = "Say 'Vending machine simulation is ready' in three words."
    print(f"Sending prompt: {prompt}")
    
    response = client.complete(prompt)
    
    if response:
        print(f"Response from LLM: {response}")
    else:
        print("Failed to get a response from OpenRouter.")

if __name__ == "__main__":
    test_connectivity()
