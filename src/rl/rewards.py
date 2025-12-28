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

def extract_json_action(text: str) -> Union[Dict[str, Any], None]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

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
    """Reward function that checks if output contains valid JSON with an action key."""
    rewards = []
    for completion in completions:
        response_text = completion if isinstance(completion, str) else completion[0]["content"]
        try:
            action_json = extract_json_action(response_text)
            if isinstance(action_json, dict) and "action" in action_json:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def profit_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward function that simulates the vending machine over multiple weeks and returns profit.
    Optimized: Batches generations across completions to significantly speed up training.
    """
    num_weeks_to_sim = kwargs.get("num_weeks", 5)
    
    # GRPO passes extra dataset columns as kwargs (lists matching the batch)
    initial_states = kwargs.get("initial_state")
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    device = next(model.parameters()).device if model else "cpu"

    num_completions = len(completions)
    print(f"[REWARD] Processing {num_completions} completions (num_weeks={num_weeks_to_sim}) in PARALLEL...", flush=True)

    # 1. Setup Phase: Initialize machines and state for each completion
    active_sims = [] # List of dicts Tracking state for each completion
    final_rewards = [0.0] * num_completions
    
    for i in range(num_completions):
        try:
            response_text = completions[i] if isinstance(completions[i], str) else completions[i][0]["content"]
            action_json = extract_json_action(response_text)

            if not isinstance(action_json, dict):
                # Provide gradient signal based on partial progress
                if '{' in response_text:
                    print(f"[REWARD] C{i} - Invalid JSON (partial structure). Penalty.", flush=True)
                    final_rewards[i] = -1.0  # Tried JSON, partial credit
                else:
                    print(f"[REWARD] C{i} - No JSON structure. Penalty.", flush=True)
                    final_rewards[i] = -2.0  # No JSON structure at all
                active_sims.append(None)
                continue

            # Initialize Machines
            basic_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
            llm_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
            
            start_week = 1
            if initial_states and i < len(initial_states):
                state = initial_states[i]
                start_week = state.get("week_num", 1)
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

            # Initial Action Validation
            valid_action = False
            action = action_json.get("action")
            if action == "change_price":
                params = action_json.get("parameters")
                if isinstance(params, dict):
                    p_name = params.get("product_name")
                    new_price = params.get("new_price")
                    if p_name and new_price is not None:
                        prod = llm_machine.products.get(p_name)
                        if prod:
                            try:
                                prod.price = float(new_price)
                                valid_action = True
                            except: pass
            elif action == "next_week":
                 valid_action = True

            if not valid_action:
                 print(f"[REWARD] C{i} - Invalid Action. Penalty.", flush=True)
                 final_rewards[i] = -1.0
                 active_sims.append(None)
                 continue

            tools = VendingMachineTools({"BasicMachine": basic_machine, "LLMMachine": llm_machine})
            active_sims.append({
                "basic_machine": basic_machine,
                "llm_machine": llm_machine,
                "tools": tools,
                "start_week": start_week,
                "current_action": action, # we already applied it
                "is_active": True
            })

        except Exception as e:
            print(f"Error initializing C{i}: {e}")
            final_rewards[i] = -5.0
            active_sims.append(None)

    # 2. Simulation Loop (by week)
    # Temporarily enable cache and disable gradient checkpointing for generation
    # GRPO disables cache by default when training with gradient checkpointing
    original_use_cache = model.config.use_cache
    
    for week_offset in range(1, num_weeks_to_sim + 1):
        prompts_to_batch = []
        indices_to_batch = []
        
        # Part A: Run simulation for the week
        for i, sim in enumerate(active_sims):
            if sim is None or not sim["is_active"]:
                continue
            
            current_week = sim["start_week"] + week_offset - 1
            results, stockout_events = simulate_competitive_week([sim["basic_machine"], sim["llm_machine"]], current_week)
            
            # Part B: If not last week, prepare for next action generation
            if week_offset < num_weeks_to_sim:
                market_data = sim["tools"].get_market_data()
                event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
                
                next_prompt = f"""
{SYSTEM_PROMPT}

--- WEEK {current_week} RESULTS ---
Stockout Events:
{event_feedback}

Current Market Data:
{json.dumps(market_data, indent=2)}

Machines are being refilled NOW. What adjustments will you make for Week {current_week+1}?
<thought>
"""
                prompts_to_batch.append(next_prompt)
                indices_to_batch.append(i)

        if prompts_to_batch and model and tokenizer:
            print(f"[REWARD] W{week_offset} - Generating actions for {len(prompts_to_batch)} sims...", flush=True)
            
            # Switch to generation mode: enable cache, disable checkpointing, use eval
            original_training_mode = model.training
            model.eval()
            model.config.use_cache = True
            model.gradient_checkpointing_disable()
            
            inputs = tokenizer(prompts_to_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Restore training mode
            model.gradient_checkpointing_enable()
            model.config.use_cache = original_use_cache
            if original_training_mode:
                model.train()
            
            if device == "mps":
                torch.mps.empty_cache()

            # Process outputs
            for j, completion_idx in enumerate(indices_to_batch):
                response_text = tokenizer.decode(outputs[j][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                next_action_json = extract_json_action(response_text)
                sim = active_sims[completion_idx]
                
                if isinstance(next_action_json, dict):
                    action = next_action_json.get("action")
                    if action == "change_price":
                        params = next_action_json.get("parameters")
                        if isinstance(params, dict):
                            p_name = params.get("product_name")
                            new_price = params.get("new_price")
                            if p_name and new_price is not None:
                                prod = sim["llm_machine"].products.get(p_name)
                                if prod:
                                    try:
                                        prod.price = float(new_price)
                                    except: pass
                    elif action == "next_week":
                        pass
                else:
                    # If model fails to produce valid JSON mid-sim, it stays with last action
                    # or we could penalize it. For now, just continue.
                    pass

    # 3. Finalization Phase: Calculate rewards
    for i, sim in enumerate(active_sims):
        if sim is None:
            continue
        profit = sim["llm_machine"].calculate_profit_loss()
        normalized_profit = profit / num_weeks_to_sim
        final_rewards[i] = normalized_profit

    print(f"[REWARD] Parallel processing DONE. Means: {sum(final_rewards)/len(final_rewards):.2f}", flush=True)
    return final_rewards

    return rewards
