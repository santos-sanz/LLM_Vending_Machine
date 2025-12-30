import re
import json
import sys
import numpy as np
import torch
from typing import List, Dict, Any, Union
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.tools import VendingMachineTools
from src.config import PRODUCT_CONFIGS, MAINTENANCE_COST
from src.rl.data_utils import SYSTEM_PROMPT


def extract_all_json_actions(text: str) -> List[Dict[str, Any]]:
    """Extract ALL JSON action blocks from the text.
    
    The model may output multiple JSON blocks, one per line:
    {"action": "change_price", "parameters": {...}}
    {"action": "change_price", "parameters": {...}}
    {"action": "next_week"}
    
    Returns a list of parsed action dictionaries.
    """
    actions = []
    # Find all JSON object patterns (greedy match for each {...})
    # Using a more robust pattern that handles nested objects
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "action" in parsed:
                actions.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return actions


def xml_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward for including both <thought> and </thought> tags."""
    rewards = []
    for completion in completions:
        response_text = completion if isinstance(completion, str) else completion[0]["content"]
        if "<thought>" in response_text and "</thought>" in response_text:
            rewards.append(0.5)
        elif "<thought>" in response_text or "</thought>" in response_text:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward function that checks if output contains valid JSON actions.
    
    Now supports multiple JSON blocks. Rewards based on:
    - Having at least one valid action: 0.5
    - Having a next_week action: +0.5 (total 1.0)
    """
    rewards = []
    for completion in completions:
        response_text = completion if isinstance(completion, str) else completion[0]["content"]
        actions = extract_all_json_actions(response_text)
        
        if not actions:
            rewards.append(0.0)
        else:
            has_next_week = any(a.get("action") == "next_week" for a in actions)
            if has_next_week:
                rewards.append(1.0)
            else:
                rewards.append(0.5)  # Partial credit for valid actions but no next_week
    return rewards


def profit_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward function that applies all price changes and simulates ONE week.
    
    Simplified flow:
    1. Extract ALL JSON actions from the completion
    2. Apply all valid change_price actions to LLMMachine
    3. Simulate ONE week of competition
    4. Return profit as reward
    
    No multi-week simulation loop - the model sees one snapshot and responds.
    """
    # GRPO passes extra dataset columns as kwargs (lists matching the batch)
    initial_states = kwargs.get("initial_state")

    num_completions = len(completions)
    print(f"[REWARD] Processing {num_completions} completions...", flush=True)

    final_rewards = [0.0] * num_completions
    
    for i in range(num_completions):
        try:
            response_text = completions[i] if isinstance(completions[i], str) else completions[i][0]["content"]
            actions = extract_all_json_actions(response_text)

            if not actions:
                # Provide gradient signal based on partial progress
                if '{' in response_text:
                    print(f"[REWARD] C{i} - Invalid JSON (partial structure). Penalty.", flush=True)
                    final_rewards[i] = -1.0  # Tried JSON, partial credit
                else:
                    print(f"[REWARD] C{i} - No JSON structure. Penalty.", flush=True)
                    final_rewards[i] = -2.0  # No JSON structure at all
                continue

            # Initialize Machines
            basic_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
            llm_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
            
            current_week = 1
            if initial_states and i < len(initial_states):
                state = initial_states[i]
                current_week = state.get("week_num", 1)
                market_data = state.get("market_data", {})
                
                llm_data = market_data.get("LLMMachine", {})
                llm_machine.cash = llm_data.get("cash", 0.0)
                llm_machine.initial_investment = 0.0
                
                basic_data = market_data.get("BasicMachine", {})
                basic_machine.cash = basic_data.get("cash", 0.0)
                basic_machine.initial_investment = 0.0

                for config in PRODUCT_CONFIGS:
                    p_name = config["name"]
                    # LLM Product
                    lp_state = llm_data.get("products", {}).get(p_name, {})
                    lp = Product(**config)
                    lp.price = lp_state.get("price", lp.price)
                    lp.stock = lp_state.get("stock", lp.stock)
                    llm_machine.add_product(lp)
                    # Basic Product
                    bp_state = basic_data.get("products", {}).get(p_name, {})
                    bp = Product(**config)
                    bp.price = bp_state.get("price", bp.price)
                    bp.stock = bp_state.get("stock", bp.stock)
                    basic_machine.add_product(bp)
            else:
                for config in PRODUCT_CONFIGS:
                    basic_machine.add_product(Product(**config))
                    llm_machine.add_product(Product(**config))

            # Process ALL actions from the completion
            price_changes_applied = 0
            duplicate_changes = 0  # Track duplicate price changes (penalized)
            products_already_changed = set()  # Track which products have been changed
            has_next_week = False
            
            for action_json in actions:
                action = action_json.get("action")
                
                if action == "change_price":
                    params = action_json.get("parameters")
                    if isinstance(params, dict):
                        p_name = params.get("product_name")
                        new_price = params.get("new_price")
                        if p_name and new_price is not None:
                            # Case-insensitive product matching
                            prod = None
                            p_name_lower = p_name.lower().strip()
                            for prod_name, product in llm_machine.products.items():
                                if prod_name.lower() == p_name_lower:
                                    prod = product
                                    p_name_lower = prod_name.lower()  # Normalize to actual name
                                    break
                            
                            if prod:
                                # Check if this product was already changed
                                if p_name_lower in products_already_changed:
                                    duplicate_changes += 1
                                else:
                                    try:
                                        old_price = prod.price
                                        prod.price = float(new_price)
                                        price_changes_applied += 1
                                        products_already_changed.add(p_name_lower)
                                    except:
                                        pass
                                    
                elif action == "next_week":
                    has_next_week = True

            # Log what was processed
            if price_changes_applied > 0 or has_next_week:
                dup_msg = f", duplicates={duplicate_changes}" if duplicate_changes > 0 else ""
                print(f"[REWARD] C{i} - Applied {price_changes_applied} price changes{dup_msg}, next_week={has_next_week}", flush=True)
            else:
                print(f"[REWARD] C{i} - No valid actions applied. Actions found: {len(actions)}", flush=True)
                final_rewards[i] = -0.5  # Has JSON but no valid actions
                continue

            # Simulate ONE week with the new prices
            results, stockout_events = simulate_competitive_week([basic_machine, llm_machine], current_week)
            
            # Calculate profit
            profit = llm_machine.calculate_profit_loss()
            
            # Normalize profit reward (e.g., 100.0 profit -> 1.0 reward)
            # This keeps training stable and prevents NaNs/gradient explosions
            scaled_profit = profit / 100.0
            
            # Bonus for making price changes (encourages exploration)
            exploration_bonus = min(price_changes_applied * 0.1, 0.4)  # Up to 0.4 bonus
            
            # Penalty for duplicate changes (indecisive behavior)
            duplicate_penalty = duplicate_changes * 0.2
            
            raw_reward = scaled_profit + exploration_bonus - duplicate_penalty
            
            # Clamp reward to prevent extremes that crash training
            final_rewards[i] = max(-2.0, min(5.0, raw_reward))

        except Exception as e:
            print(f"[REWARD] Error processing C{i}: {e}", flush=True)
            final_rewards[i] = -5.0

    mean_reward = sum(final_rewards) / len(final_rewards) if final_rewards else 0
    print(f"[REWARD] Processing DONE. Mean reward: {mean_reward:.2f}", flush=True)
    return final_rewards
