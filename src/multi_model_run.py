import json
import time
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.client import OpenRouterClient
from src.llm.tools import VendingMachineTools
from src.config import (
    MODEL_A, 
    MODEL_B, 
    PRODUCT_CONFIGS, 
    MAINTENANCE_COST
)
from src.utils.helpers import plot_multi_profits, extract_json_objects

def run_llm_turn(machine_name, model_name, client, tools, week, market_data, event_feedback, system_prompt):
    print(f"\n--- {machine_name} ({model_name}) Turn ---")
    
    while True:
        prompt = f"""
        --- WEEK {week} RESULTS ---
        Stockout Events:
        {event_feedback}
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        You are controlling '{machine_name}'. 
        Machines are being refilled NOW. What adjustments will you make for Week {week+1}?
        """
        
        llm_response = client.complete(prompt, model=model_name, system_prompt=system_prompt)
        if not llm_response:
            print(f"No response from {model_name}.")
            break
            
        print(f"\nResponse from {model_name}:\n{llm_response}")

        commands = extract_json_objects(llm_response)
        if not commands:
            print(f"No valid commands found in {model_name} response.")
            break

        should_proceed = False
        for command in commands:
            try:
                action = command.get("action")
                params = command.get("parameters", {})
                
                if action == "change_price" and (params.get("machine_name") == machine_name or (isinstance(params, list) and any(p.get("machine_name") == machine_name for p in params))):
                    if isinstance(params, list):
                        for p in params:
                            if p.get("machine_name") == machine_name:
                                result = tools.change_price(**p)
                                print(f"Tool Execution Result: {result}")
                    else:
                        result = tools.change_price(**params)
                        print(f"Tool Execution Result: {result}")
                    market_data = tools.get_market_data()
                elif action == "next_week":
                    print(f"{machine_name} decided to proceed.")
                    should_proceed = True
                    break
                else:
                    print(f"Skipping invalid or unauthorized command: {action}")
            except Exception as e:
                print(f"Error executing command: {e}")
        
        if should_proceed:
            break
            
        if len(commands) > 0:
            print(f"{machine_name} turn completed (processed {len(commands)} commands).")
            break

def main():
    print("=== Script 3: Multi-Model Vending Machine Competition ===\n")

    # 1. Initialize Machines
    basic_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
    llm_a_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
    llm_b_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)

    for config in PRODUCT_CONFIGS:
        basic_machine.add_product(Product(**config))
        llm_a_machine.add_product(Product(**config))
        llm_b_machine.add_product(Product(**config))

    machines = {
        "BasicMachine": basic_machine,
        "LLMMachine_A": llm_a_machine,
        "LLMMachine_B": llm_b_machine
    }

    # Data collection
    weeks_list = []
    basic_profits = []
    model_a_profits = []
    model_b_profits = []

    # 2. Initialize LLM and Tools
    llm_client = OpenRouterClient()
    tools = VendingMachineTools(machines)

    system_prompt = f"""
    You are a Strategic Business Manager. You are in direct competition with two other machines.
    All machines share the same client pool. Clients choose the product with the highest 'purchase_likelihood'.
    
    IMPORTANT RULES:
    1. AT THE START OF EACH WEEK, YOUR MACHINE IS REFILLED TO MAX CAPACITY.
    2. 'BasicMachine' NEVER changes its prices.
    3. Your goal is to maximize NET PROFIT.
    
    STRATEGIC ADVICE:
    - If you sell out early, your price is TOO LOW.
    - The "Sweet Spot" is to sell out on the very last client of the week.
    
    AVAILABLE TOOLS (Respond only in JSON):
    1. {{"action": "change_price", "parameters": {{"machine_name": "YOUR_MACHINE_NAME", "product_name": "...", "new_price": ...}}}}
    2. {{"action": "next_week"}} - to proceed.
    
    Provide your reasoning first, then the JSON block.
    """

    # 3. Simulation Loop
    num_weeks = 10
    for week in range(1, num_weeks + 1):
        # 1. Simulate the week
        results, stockout_events = simulate_competitive_week([basic_machine, llm_a_machine, llm_b_machine], week)

        # 2. Log profits
        weeks_list.append(week)
        basic_profits.append(basic_machine.calculate_profit_loss())
        model_a_profits.append(llm_a_machine.calculate_profit_loss())
        model_b_profits.append(llm_b_machine.calculate_profit_loss())

        # 3. Get data for LLMs
        market_data = tools.get_market_data()
        event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
        
        # 4. LLM Turns
        run_llm_turn("LLMMachine_A", MODEL_A, llm_client, tools, week, market_data, event_feedback, system_prompt)
        time.sleep(5) # Small delay to avoid hitting rate limits too quickly
        run_llm_turn("LLMMachine_B", MODEL_B, llm_client, tools, week, market_data, event_feedback, system_prompt)

    print("\n=== Multi-Model Competition Finished ===")
    print(f"BasicMachine Final Profit: {basic_machine.calculate_profit_loss():.2f}")
    print(f"Model A ({MODEL_A}) Final Profit: {llm_a_machine.calculate_profit_loss():.2f}")
    print(f"Model B ({MODEL_B}) Final Profit: {llm_b_machine.calculate_profit_loss():.2f}")

    # Generate Graph
    from src.config import IMAGES_DIR
    import os
    from datetime import datetime
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_a = MODEL_A.replace("/", "_").replace(":", "_")
    safe_b = MODEL_B.replace("/", "_").replace(":", "_")
    filename = f"multi_profit_comparison_{safe_a}_vs_{safe_b}_{timestamp}.png"
    graph_path = os.path.join(IMAGES_DIR, filename)
    
    plot_multi_profits(weeks_list, basic_profits, model_a_profits, model_b_profits, MODEL_A, MODEL_B, graph_path)

if __name__ == "__main__":
    main()
