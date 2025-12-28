import os
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Configuration
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
CHECKPOINT_PATH = "outputs/SmolLM3-3B_rl_vending_local/checkpoint-300"
OUTPUT_DIR = "outputs/SmolLM3-3B_rl_vending_local"
HF_ACCOUNT = os.getenv("HUGGINGFACE_ACCOUNT")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def main():
    if not HF_ACCOUNT or not HF_TOKEN:
        print("❌ HUGGINGFACE_ACCOUNT or HUGGINGFACE_TOKEN not found in .env")
        return

    print(f"Logging in to Hugging Face...")
    login(token=HF_TOKEN)

    repo_name = f"{HF_ACCOUNT}/vending-machine-rl-model"
    print(f"Target repository: {repo_name}")

    print(f"Loading tokenizer from {CHECKPOINT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

    print(f"Loading base model: {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="cpu" # Use CPU for simple upload to save memory/time
    )

    print(f"Loading LoRA adapter from {CHECKPOINT_PATH}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)

    print(f"Pushing model (adapter) to Hugging Face Hub...")
    model.push_to_hub(
        repo_name,
        token=HF_TOKEN,
        commit_message="Upload RL-trained vending machine model (Step 300)"
    )

    print(f"Pushing tokenizer to Hugging Face Hub...")
    tokenizer.push_to_hub(
        repo_name,
        token=HF_TOKEN,
        commit_message="Upload tokenizer for RL-trained vending machine model"
    )

    print(f"Pushing README.md (Model Card) to Hugging Face Hub...")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=os.path.join(OUTPUT_DIR, "README.md"),
        path_in_repo="README.md",
        repo_id=repo_name,
        token=HF_TOKEN
    )

    print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    main()
