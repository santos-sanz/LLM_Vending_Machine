import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Model Configuration ---
# OTHERS MODELS Qwen/Qwen3-0.6B HuggingFaceTB/SmolLM3-3B
MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/Qwen3-0.6B_rl_vending_local"
MAX_SEQ_LENGTH = 1536

# --- LoRA Configuration ---
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Training Configuration ---
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GENERATIONS = 4  # Reduced from 8 for stability on MPS
MAX_COMPLETION_LENGTH = 512  # Reduced for MPS memory constraints
LEARNING_RATE = 1e-5  # Further reduced for stability
WARMUP_STEPS = 20
LOGGING_STEPS = 1

# --- Paths ---
HUGGINGFACE_ACCOUNT = os.getenv("HUGGINGFACE_ACCOUNT")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
