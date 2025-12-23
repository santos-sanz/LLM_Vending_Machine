import json
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.client import OpenRouterClient
from src.llm.tools import VendingMachineTools
from src.config import DEFAULT_MODEL

def plot_profits(weeks, basic_profits, llm_profits, model_name, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(weeks, basic_profits, label='Basic Vending Machine', marker='o', linestyle='--', color='blue')
    plt.plot(weeks, llm_profits, label=f'LLM Vending Machine ({model_name})', marker='s', linestyle='-', color='green')
    
    plt.title(f'Weekly Net Profit Comparison ({model_name})')
    plt.xlabel('Week')
    plt.ylabel('Net Profit/Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(filename)
    print(f"\nProfit graph saved as {filename}")
    plt.close()

def main():
    print("=== Script 2: Competitive LLM Vending Machine Simulation ===\n")

    # 1. Initialize Machines
    product_configs = [
        # (name, price, cost, stock, max_stock, base_likelihood, price_sensitivity)
        ("Cola", 2.5, 0.8, 20, 20, 0.8, 0.1),
        ("Water", 1.5, 0.3, 30, 30, 0.9, 0.2),
        ("Chips", 2.0, 0.6, 15, 15, 0.7, 0.15)
    ]

    basic_machine = VendingMachine(initial_cash=0, maintenance_cost=50.0, initial_investment=0.0)
    llm_machine = VendingMachine(initial_cash=0, maintenance_cost=50.0, initial_investment=0.0)

    for config in product_configs:
        basic_machine.add_product(Product(*config))
        # Important: Create separate Product instances for each machine
        llm_machine.add_product(Product(*config))

    machines = {
        "BasicMachine": basic_machine,
        "LLMMachine": llm_machine
    }

    # Data collection for graphing
    weeks_list = []
    basic_profits = []
    llm_profits = []
    last_week_events = []

    # 2. Initialize LLM and Tools
    llm_client = OpenRouterClient()
    tools = VendingMachineTools(machines)

    system_prompt = f"""
    You are a Strategic Business Manager for 'LLMMachine'. You are in direct competition with 'BasicMachine'.
    Both machines share the same client pool. Clients choose the product with the highest 'purchase_likelihood'.
    
    IMPORTANT RULES:
    1. AT THE START OF EACH WEEK, YOUR MACHINE IS REFILLED TO MAX CAPACITY.
    2. 'BasicMachine' NEVER changes its prices (Cola: 2.5, Water: 1.5, Chips: 2.0).
    3. Your goal is to maximize NET PROFIT.
    
    STRATEGIC ADVICE:
    - If you sell out early (e.g., at client 50 out of 150), your price is TOO LOW. You are losing potential margin!
    - The "Sweet Spot" is to sell out on the very last client of the week.
    - If you don't sell out, your price might be too high, or the competitor is stealing your share.
    
    AVAILABLE TOOLS (Respond only in JSON):
    1. {{"action": "change_price", "parameters": {{"machine_name": "LLMMachine", "product_name": "...", "new_price": ...}}}}
    2. {{"action": "next_week"}} - to proceed.
    
    Provide your reasoning first, then the JSON block.
    """

    # 3. Simulation Loop
    num_weeks = 52
    for week in range(1, num_weeks + 1):
        # 1. Simulate the week
        results, stockout_events = simulate_competitive_week([basic_machine, llm_machine], week)

        # 2. Log profits
        weeks_list.append(week)
        basic_profits.append(basic_machine.calculate_profit_loss())
        llm_profits.append(llm_machine.calculate_profit_loss())

        # 3. Get data for LLM
        market_data = tools.get_market_data()
        
        # Prepare event feedback
        event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
        print(f"\nMarket Data for LLM:\n{json.dumps(market_data, indent=2)}")

        # 4. LLM Decision Loop
        while True:
            prompt = f"""
            --- WEEK {week} RESULTS ---
            Stockout Events:
            {event_feedback}
            
            Current Market Data:
            {json.dumps(market_data, indent=2)}
            
            Machines are being refilled NOW. What adjustments will you make for Week {week+1}?
            """
            
            llm_response = llm_client.complete(prompt, system_prompt=system_prompt)
            print(f"\nLLM Response: {llm_response}")

            try:
                # Find the JSON part in the response
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    command = json.loads(llm_response[start_idx:end_idx])
                    
                    if command.get("action") == "change_price":
                        params = command.get("parameters", {})
                        result = tools.change_price(**params)
                        print(f"Tool Execution Result: {result}")
                        market_data = tools.get_market_data()
                    elif command.get("action") == "next_week":
                        print("LLM decided to proceed to the next week.")
                        break
                    else:
                        break
                else:
                    break
            except Exception as e:
                print(f"Error parsing LLM command: {e}")
                break

    print("\n=== Competition Finished ===")
    print(f"BasicMachine Final Profit: {basic_machine.calculate_profit_loss():.2f}")
    print(f"LLMMachine Final Profit: {llm_machine.calculate_profit_loss():.2f}")

    # Generate Unique Filenames and Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_filename = f"profit_comparison_{timestamp}.png"
    
    # Generate Graph
    plot_profits(weeks_list, basic_profits, llm_profits, DEFAULT_MODEL, graph_filename)

    # 4. Persistence: Store in CSV
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "simulation_history.csv")
    
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "num_weeks", "basic_profit", "llm_profit", "graph_path"])
        
        writer.writerow([
            timestamp,
            DEFAULT_MODEL,
            num_weeks,
            f"{basic_machine.calculate_profit_loss():.2f}",
            f"{llm_machine.calculate_profit_loss():.2f}",
            graph_filename
        ])
    
    print(f"Results appended to {csv_path}")

if __name__ == "__main__":
    main()
