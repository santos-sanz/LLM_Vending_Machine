import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- General Simulation Settings ---
NUM_WEEKS = 52
INITIAL_CASH = 500.0
MAINTENANCE_COST = 5.0

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
CLIENT_LAMBDA = 25
# Range for uniform distribution fallback
CLIENT_UNIFORM_RANGE = (0, 50)

# --- LLM Settings (nan API, OpenAI-compatible) ---
NAN_API_KEY = os.getenv("NAN_API_KEY")
NAN_BASE_URL = os.getenv("NAN_BASE_URL") or "https://api.nan.builders/v1"
# --- LLM Settings (OpenRouter) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# The .env value serves as the primary choice; a hardcoded model as a fallback.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or "deepseek-v4-flash"

# Chat-capable models on the nan API (audio/image/embedding models such as
# kokoro, whisper, rerank, qwen3-embedding, flux-2-klein and qwen-image-2.1
# are excluded because they cannot act as agents). gemma4 is excluded from
# benchmarking: its reasoning chains overflow the API gateway timeout (524)
# and its server-side concurrency cap makes full runs unreliable.
NAN_CHAT_MODELS = [
    "deepseek-v4-flash",
    "qwen3.8-flash",
    "glm5.3-flash",
    "qwen3.6",
]

# Top-5 weekly "decisions" models on OpenRouter (output_modalities=decisions,
# ordered by https://openrouter.ai/models?order=top-weekly). These models do
# not accept chat/completions; they are driven through POST /api/alpha/decisions.
OPENROUTER_DECISION_MODELS = [
    "typesafe/jev-1.13",
    "perplexity/pplx-decider-v1.1-27b",
    "openai/gpt-6-luna-decisions",
    "cloudflare/clef-flash",
    "inception/mercury-decide:free",
]

# Per-model generation overrides (applied by agentic_run via ChatOpenAI kwargs).
# Several nan models are "reasoning" models that otherwise generate very long
# hidden reasoning chains, which makes every call take minutes and can exceed
# the API gateway timeout (HTTP 524). These controls bound the reasoning so
# calls finish in seconds while keeping valid JSON tool output.
MODEL_GENERATION_OVERRIDES = {
    "deepseek-v4-flash": {"extra_body": {"enable_thinking": False}},
    "qwen3.8-flash": {"extra_body": {"reasoning_effort": "low"}},
    "qwen3.6": {"extra_body": {"reasoning_effort": "low"}},
    "glm5.3-flash": {"extra_body": {"reasoning_effort": "low"}},
    "gemma4": {"max_tokens": 4000},
}


def resolve_llm_config(model):
    """Return (api_key, base_url) for a model, routing nan vs OpenRouter models."""
    if model in OPENROUTER_DECISION_MODELS:
        return OPENROUTER_API_KEY, OPENROUTER_BASE_URL
    return NAN_API_KEY, NAN_BASE_URL

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
