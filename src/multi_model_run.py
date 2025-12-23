import json
import re
import matplotlib.pyplot as plt
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.client import OpenRouterClient
from src.llm.tools import VendingMachineTools
from src.config import MODEL_A, MODEL_B

def plot_multi_profits(weeks, basic_profits, model_a_profits, model_b_profits, model_a_name, model_b_name):
    plt.figure(figsize=(12, 7))
    plt.plot(weeks, basic_profits, label='Basic Vending Machine', marker='o', linestyle='--', color='blue')
    plt.plot(weeks, model_a_profits, label=f'Model A ({model_a_name})', marker='s', linestyle='-', color='green')
    plt.plot(weeks, model_b_profits, label=f'Model B ({model_b_name})', marker='^', linestyle='-', color='red')
    
    plt.title('Multi-Model Weekly Net Profit Comparison')
    plt.xlabel('Week')
    plt.ylabel('Net Profit/Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    filename = 'multi_profit_comparison.png'
    plt.savefig(filename)
    print(f"\nMulti-model profit graph saved as {filename}")
    plt.close()

def run_llm_turn(machine_name, model_name, client, tools, week, market_data, event_feedback, system_prompt):
    print(f"\n--- {machine_name} ({model_name}) Turn ---")
    
    # Robust extraction of JSON
    def extract_json_objects(text):
        if not text:
            return []
        
        # 1. Clean up "thinking" tags if present (common in DeepSeek and others)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 2. Try to find content within markdown blocks
        code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            objs = []
            for block in code_blocks:
                try:
                    loaded = json.loads(block)
                    if isinstance(loaded, list):
                        objs.extend(loaded)
                    else:
                        objs.append(loaded)
                except json.JSONDecodeError:
                    # If the block itself isn't a single object/list, try searching for objects inside it
                    objs.extend(extract_json_objects(block))
            if objs:
                return objs

        # 3. Fallback: Search for anything that looks like a JSON object or array
        # This version tries to find ALL objects by searching forward
        results = []
        i = 0
        while i < len(text):
            if text[i] in '{[':
                found_at_this_pos = False
                for j in range(len(text), i, -1):
                    try:
                        loaded = json.loads(text[i:j])
                        if isinstance(loaded, list):
                            results.extend(loaded)
                        else:
                            results.append(loaded)
                        i = j # Skip to the end of this object
                        found_at_this_pos = True
                        break
                    except json.JSONDecodeError:
                        continue
                if not found_at_this_pos:
                    i += 1
            else:
                i += 1
        return results

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
                
                if action == "change_price" and params.get("machine_name") == machine_name:
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
            
        # If the model didn't explicitly say next_week in its list of commands, 
        # but it performed actions, we might want to let it continue or just stop.
        # Most simple models will just list changes and stop.
        # To avoid infinite loops where the model keeps repeating the same changes,
        # we'll break if it provided a list of commands but no 'next_week'.
        if len(commands) > 0:
            print(f"{machine_name} turn completed (processed {len(commands)} commands).")
            break

def main():
    print("=== Script 3: Multi-Model Vending Machine Competition ===\n")

    # 1. Initialize Machines
    product_configs = [
        ("Soda", 2.50, 1.00, 20, 20, 0.725, 0.05),
        ("Chips", 1.75, 0.75, 30, 30, 0.54, 0.08),
        ("Candy Bar", 1.25, 0.50, 40, 40, 0.75, 0.04),
        ("Water", 1.00, 0.30, 50, 50, 0.83, 0.03)
    ]

    basic_machine = VendingMachine(initial_cash=0, maintenance_cost=20.0, initial_investment=0.0)
    llm_a_machine = VendingMachine(initial_cash=0, maintenance_cost=20.0, initial_investment=0.0)
    llm_b_machine = VendingMachine(initial_cash=0, maintenance_cost=20.0, initial_investment=0.0)

    for config in product_configs:
        basic_machine.add_product(Product(*config))
        llm_a_machine.add_product(Product(*config))
        llm_b_machine.add_product(Product(*config))

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
        import time
        run_llm_turn("LLMMachine_A", MODEL_A, llm_client, tools, week, market_data, event_feedback, system_prompt)
        time.sleep(5) # Small delay to avoid hitting rate limits too quickly
        run_llm_turn("LLMMachine_B", MODEL_B, llm_client, tools, week, market_data, event_feedback, system_prompt)

    print("\n=== Multi-Model Competition Finished ===")
    print(f"BasicMachine Final Profit: {basic_machine.calculate_profit_loss():.2f}")
    print(f"Model A ({MODEL_A}) Final Profit: {llm_a_machine.calculate_profit_loss():.2f}")
    print(f"Model B ({MODEL_B}) Final Profit: {llm_b_machine.calculate_profit_loss():.2f}")

    # Generate Graph
    plot_multi_profits(weeks_list, basic_profits, model_a_profits, model_b_profits, MODEL_A, MODEL_B)

if __name__ == "__main__":
    main()
