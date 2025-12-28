import random
import json
import numpy as np
from typing import Dict, Any
from src.config import PRODUCT_CONFIGS, CLIENT_LAMBDA

SYSTEM_PROMPT = """
You are a Strategic Business Manager for 'LLMMachine'. You are in direct competition with 'BasicMachine'.
Both machines share the same client pool. Clients choose products probabilistically based on their 'purchase_likelihood'.

IMPORTANT RULES:
1. AT THE START OF EACH WEEK, YOUR MACHINE IS REFILLED TO MAX CAPACITY.
2. 'BasicMachine' NEVER changes its prices.
3. Your goal is to maximize NET PROFIT.

STRATEGIC ADVICE:
- Your priority is PROFIT, not just sales volume.
- Selling out is NOT always the goal. If there is more supply than demand, you might never sell out.
- If you raise prices, you might sell fewer items, but if the margin increase covers the volume loss, your PROFIT goes up.
- BasicMachine prices are static. Use this to your advantage.
- Monitor your Weekly Profit. If a price hike increased profit, keep it or hike further.

AVAILABLE TOOLS (Respond only in JSON):
1. {"action": "change_price", "parameters": {"machine_name": "LLMMachine", "product_name": "...", "new_price": ...}}
2. {"action": "next_week"} - to proceed.

Provide your reasoning inside <thought>...</thought> tags, followed by exactly one JSON block.

Example:
<thought>
I will raise the price of Soda because it is currently cheaper than BasicMachine and we are selling out.
</thought>
{"action": "change_price", "parameters": {"product_name": "Soda", "new_price": 2.5}}
"""

def generate_random_market_data() -> Dict[str, Any]:
    """Generate market data that simulates what LLM would see during competition.
    Mimics VendingMachineTools.get_market_data() format.
    """
    basic_machine_data = {
        "cash": round(random.uniform(-100, 500), 2),
        "profit_loss": round(random.uniform(-200, 300), 2),
        "products": {}
    }
    
    llm_machine_data = {
        "cash": round(random.uniform(-100, 500), 2),
        "profit_loss": round(random.uniform(-200, 300), 2),
        "products": {}
    }
    
    for p_config in PRODUCT_CONFIGS:
        # BasicMachine always has fixed prices from config
        basic_price = p_config["price"]
        basic_stock = random.randint(0, p_config["max_stock"])
        basic_likelihood = p_config["base_likelihood"] - (basic_price * p_config["price_sensitivity"])
        
        basic_machine_data["products"][p_config["name"]] = {
            "price": basic_price,
            "cost": p_config["cost"],
            "stock": basic_stock,
            "max_stock": p_config["max_stock"],
            "likelihood": max(0.0, min(1.0, basic_likelihood))
        }
        
        # LLMMachine might have modified prices
        llm_price = round(basic_price * random.uniform(0.8, 1.3), 2)
        llm_stock = random.randint(0, p_config["max_stock"])
        llm_likelihood = p_config["base_likelihood"] - (llm_price * p_config["price_sensitivity"])
        
        llm_machine_data["products"][p_config["name"]] = {
            "price": llm_price,
            "cost": p_config["cost"],
            "stock": llm_stock,
            "max_stock": p_config["max_stock"],
            "likelihood": max(0.0, min(1.0, llm_likelihood))
        }
    
    return {
        "BasicMachine": basic_machine_data,
        "LLMMachine": llm_machine_data
    }

def generate_random_stockout_events() -> str:
    if random.random() < 0.7:
        return "No products ran out of stock this week."
    events = []
    num_events = random.randint(1, 3)
    for _ in range(num_events):
        prod = random.choice(PRODUCT_CONFIGS)["name"]
        events.append(f"Machine 0: {prod} is out of stock!")
    return "\n".join(events)

def generate_prompt(idx: int) -> Dict[str, Any]:
    market_data = generate_random_market_data()
    event_feedback = generate_random_stockout_events()
    week_num = random.randint(1, 52)

    prompt = f"""
    {SYSTEM_PROMPT}

    --- WEEK {week_num} RESULTS ---
    Stockout Events:
    {event_feedback}

    Current Market Data:
    {json.dumps(market_data, indent=2)}

    Machines are being refilled NOW. What adjustments will you make for Week {week_num+1}?
    <thought>
    """
    return {
        "prompt": prompt,
        "initial_state": {
            "market_data": market_data,
            "week_num": week_num,
            "event_feedback": event_feedback
        }
    }
