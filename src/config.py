import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- General Simulation Settings ---
NUM_WEEKS = 52
INITIAL_CASH = 500.0
MAINTENANCE_COST = 20.0

# --- Product Configurations ---
# Format: {"name": str, "price": float, "cost": float, "stock": int, "max_stock": int, "base_likelihood": float, "price_sensitivity": float}
PRODUCT_CONFIGS = [
    {"name": "Soda",      "price": 2.50, "cost": 1.00, "stock": 20, "max_stock": 20, "base_likelihood": 0.725, "price_sensitivity": 0.05},
    {"name": "Chips",     "price": 1.75, "cost": 0.75, "stock": 30, "max_stock": 30, "base_likelihood": 0.54,  "price_sensitivity": 0.08},
    {"name": "Candy Bar", "price": 1.25, "cost": 0.50, "stock": 40, "max_stock": 40, "base_likelihood": 0.75,  "price_sensitivity": 0.04},
    {"name": "Water",     "price": 1.00, "cost": 0.30, "stock": 50, "max_stock": 50, "base_likelihood": 0.83,  "price_sensitivity": 0.03}
]

# Unify both configurations to use the same product set
COMPETITION_PRODUCT_CONFIGS = PRODUCT_CONFIGS

# --- Client Traffic Settings ---
CLIENT_DISTRIBUTION = "poisson"  # Options: "poisson", "uniform"
# Lambda represents the average number of clients per week for Poisson distribution
CLIENT_LAMBDA = 12
# Range for uniform distribution fallback
CLIENT_UNIFORM_RANGE = (0, 50)

# --- LLM Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# The .env value serves as the primary choice; a hardcoded model as a fallback.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or "nvidia/nemotron-3-nano-30b-a3b:free"

# --- Local Model Settings (Ollama) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
DEFAULT_LOCAL_MODEL = os.getenv("DEFAULT_LOCAL_MODEL") or "qwen"

# --- Multi-Model Competition Settings ---
MODEL_A = "xiaomi/mimo-v2-flash:free"
MODEL_B = "mistralai/devstral-2512:free"

# --- Output Settings ---
RESULTS_DIR = "data/results"
LOGS_DIR = "data/logs"
IMAGES_DIR = "data/images"
BENCHMARKS_DIR = "data/benchmarks"
SIMULATION_RESULTS_CSV = os.path.join(RESULTS_DIR, "simulation_history.csv")
