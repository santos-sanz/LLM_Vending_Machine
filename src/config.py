import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define product configs in a list of dicts
PRODUCT_CONFIGS = [
    {"name": "Soda",      "price": 2.50, "cost": 1.00, "stock": 20, "max_stock": 20, "base_likelihood": 0.725, "price_sensitivity": 0.05},
    {"name": "Chips",     "price": 1.75, "cost": 0.75, "stock": 30, "max_stock": 30, "base_likelihood": 0.54,  "price_sensitivity": 0.08},
    {"name": "Candy Bar", "price": 1.25, "cost": 0.50, "stock": 40, "max_stock": 40, "base_likelihood": 0.75,  "price_sensitivity": 0.04},
    {"name": "Water",     "price": 1.00, "cost": 0.30, "stock": 50, "max_stock": 50, "base_likelihood": 0.83,  "price_sensitivity": 0.03}
]

# Vending Machine Settings
INITIAL_CASH = 500.0
MAINTENANCE_COST = 20.0
NUM_WEEKS = 52

# LLM Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
#DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "xiaomi/mimo-v2-flash:free")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/devstral-2512:free")

# Multi-Model Competition Settings
MODEL_A = "xiaomi/mimo-v2-flash:free"
MODEL_B = "mistralai/devstral-2512:free"
