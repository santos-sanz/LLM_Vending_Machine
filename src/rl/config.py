import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Model Configuration ---
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
OUTPUT_DIR = "outputs/SmolLM3-3B_rl_vending_local"
MAX_SEQ_LENGTH = 1536

# --- LoRA Configuration ---
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Training Configuration ---
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GENERATIONS = 8
MAX_COMPLETION_LENGTH = 256
LEARNING_RATE = 1e-4
WARMUP_STEPS = 10
LOGGING_STEPS = 1

# --- Paths ---
HUGGINGFACE_ACCOUNT = os.getenv("HUGGINGFACE_ACCOUNT")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
