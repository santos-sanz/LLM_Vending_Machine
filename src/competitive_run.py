import json
import csv
import os
from datetime import datetime
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.client import OpenRouterClient
from src.llm.tools import VendingMachineTools
from src.config import (
    DEFAULT_MODEL, 
    COMPETITION_PRODUCT_CONFIGS, 
    MAINTENANCE_COST, 
    NUM_WEEKS, 
    SIMULATION_RESULTS_CSV
)
from src.utils.helpers import plot_profits, extract_json_objects

def run_competition(model_name=None, verbose=True, num_weeks=None, save_plot=True, record_history=True):
    """
    Runs a competitive simulation between BasicMachine and LLMMachine.
    If model_name is provided, it overrides the DEFAULT_MODEL from config.
    If num_weeks is provided, it overrides the global NUM_WEEKS.
    If save_plot is False, graph generation is skipped.
    If record_history is False, global simulation history is not updated.
    """
    target_model = model_name if model_name else DEFAULT_MODEL
    sim_weeks = num_weeks if num_weeks is not None else NUM_WEEKS
    
    if verbose:
        print(f"--- Starting Competition: BasicMachine vs {target_model} ({sim_weeks} weeks) ---\n")

    # 1. Initialize Machines
    basic_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
    llm_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)

    for config in COMPETITION_PRODUCT_CONFIGS:
        basic_machine.add_product(Product(**config))
        # Important: Create separate Product instances for each machine
        llm_machine.add_product(Product(**config))

    machines = {
        "BasicMachine": basic_machine,
        "LLMMachine": llm_machine
    }

    # Data collection for graphing and logging
    weeks_list = []
    basic_profits = []
    llm_profits = []
    
    # New detailed logs
    weekly_stats = []

    # 2. Initialize LLM and Tools
    llm_client = OpenRouterClient()
    tools = VendingMachineTools(machines)

    system_prompt = f"""
    You are a Strategic Business Manager for 'LLMMachine'. You are in direct competition with 'BasicMachine'.
    Both machines share the same client pool. Clients choose the product with the highest 'purchase_likelihood'.

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
    1. {{"action": "change_price", "parameters": {{"machine_name": "LLMMachine", "product_name": "...", "new_price": ...}}}}
    2. {{"action": "next_week"}} - to proceed.
    
    Provide your reasoning first, then the JSON block.
    """

    # 3. Simulation Loop
    for week in range(1, sim_weeks + 1):
        # 1. Simulate the week
        results, stockout_events = simulate_competitive_week([basic_machine, llm_machine], week)

        # 2. Log profits and calculate averages
        weeks_list.append(week)
        
        b_profit = basic_machine.calculate_profit_loss()
        l_profit = llm_machine.calculate_profit_loss()
        
        basic_profits.append(b_profit)
        llm_profits.append(l_profit)

        # Calculate averages for logging
        b_avg_price = sum(p.price for p in basic_machine.products.values()) / len(basic_machine.products)
        l_avg_price = sum(p.price for p in llm_machine.products.values()) / len(llm_machine.products)
        
        b_avg_stock = sum(p.stock for p in basic_machine.products.values()) / len(basic_machine.products)
        l_avg_stock = sum(p.stock for p in llm_machine.products.values()) / len(llm_machine.products)

        weekly_stats.append({
            "week": week,
            "basic_profit": f"{b_profit:.2f}",
            "llm_profit": f"{l_profit:.2f}",
            "basic_avg_price": f"{b_avg_price:.2f}",
            "llm_avg_price": f"{l_avg_price:.2f}",
            "basic_avg_stock": f"{b_avg_stock:.2f}",
            "llm_avg_stock": f"{l_avg_stock:.2f}"
        })

        # 3. Get data for LLM
        market_data = tools.get_market_data()
        
        # Prepare event feedback
        event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
        if verbose:
            print(f"\nMarket Data for LLM:\n{json.dumps(market_data, indent=2)}")

        # 4. LLM Decision Loop
        is_last_week = (week == sim_weeks)
        while True:
            if is_last_week:
                prompt = f"""
                --- FINAL WEEK {week} RESULTS ---
                Stockout Events:
                {event_feedback}
                
                Current Market Data:
                {json.dumps(market_data, indent=2)}
                
                NOTE: This was the FINAL week of the simulation. No further actions can be taken. 
                Please provide your final analysis of the competition. 
                (Do NOT use 'next_week').
                """
            else:
                prompt = f"""
                --- WEEK {week} RESULTS ---
                Stockout Events:
                {event_feedback}
                
                Current Market Data:
                {json.dumps(market_data, indent=2)}
                
                Machines are being refilled NOW. What adjustments will you make for Week {week+1}?
                """
            
            # Note: We pass the target_model here to override the default if specified
            llm_response = llm_client.complete(prompt, model=target_model, system_prompt=system_prompt)
            if verbose:
                print(f"\nLLM Response: {llm_response}")

            if not llm_response:
                break

            # On the last week, we just want the response/analysis, no more actions
            if is_last_week:
                 break

            commands = extract_json_objects(llm_response)
            if not commands:
                break

            found_next_week = False
            for command in commands:
                if command.get("action") == "change_price":
                    params = command.get("parameters", {})
                    if isinstance(params, list):
                        for p in params:
                            try:
                                result = tools.change_price(**p)
                                if verbose:
                                    print(f"Tool Execution Result: {result}")
                            except TypeError as e:
                                print(f"  [Error] Invalid arguments for change_price: {p}. Error: {e}")
                    else:
                        try:
                            result = tools.change_price(**params)
                            if verbose:
                                print(f"Tool Execution Result: {result}")
                        except TypeError as e:
                            print(f"  [Error] Invalid arguments for change_price: {params}. Error: {e}")
                    market_data = tools.get_market_data()
                elif command.get("action") == "next_week":
                    if verbose:
                        print("LLM decided to proceed to the next week.")
                    found_next_week = True
                    break
            
            if found_next_week or commands: # If we processed commands but didn't find next_week, we still break to avoid loops
                break

    final_basic = basic_machine.calculate_profit_loss()
    final_llm = llm_machine.calculate_profit_loss()

    if verbose:
        print("\n=== Competition Finished ===")
        print(f"BasicMachine Final Profit: {final_basic:.2f}")
        print(f"LLMMachine ({target_model}) Final Profit: {final_llm:.2f}")

    # Generate Unique Filenames and Paths
    from src.config import IMAGES_DIR
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = target_model.replace("/", "_").replace(":", "_")
    graph_filename = f"profit_comparison_{safe_model_name}_{timestamp}.png"
    graph_path = os.path.join(IMAGES_DIR, graph_filename)
    
    # Generate Graph
    if save_plot:
        plot_profits(weeks_list, basic_profits, llm_profits, target_model, graph_path)
    elif verbose:
        print("Skipping individual plot generation (save_plot=False)")

    # 4. Persistence: Detailed Weekly Log
    from src.config import LOGS_DIR
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_filename = f"log_{safe_model_name}_{timestamp}.csv"
    log_path = os.path.join(LOGS_DIR, log_filename)
    
    with open(log_path, mode='w', newline='') as f:
        if weekly_stats:
            writer = csv.DictWriter(f, fieldnames=weekly_stats[0].keys())
            writer.writeheader()
            writer.writerows(weekly_stats)
    
    if verbose:
        print(f"Detailed weekly log saved as {log_path}")

    # 5. Persistence: Global History Summary
    if record_history:
        from src.config import RESULTS_DIR
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        file_exists = os.path.isfile(SIMULATION_RESULTS_CSV)
        with open(SIMULATION_RESULTS_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model", "num_weeks", "basic_profit", "llm_profit", "graph_path"])
            
            writer.writerow([
                timestamp,
                target_model,
                sim_weeks,  # FIXED: Use the actual weeks executed
                f"{final_basic:.2f}",
                f"{final_llm:.2f}",
                graph_filename
            ])
        
        if verbose:
            print(f"Results appended to {SIMULATION_RESULTS_CSV}")
    elif verbose:
        print("Skipping global history update (record_history=False)")
    
    return {
        "timestamp": timestamp,
        "model": target_model,
        "basic_profit": final_basic,
        "llm_profit": final_llm,
        "weekly_stats": weekly_stats
    }

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Competitive LLM Vending Machine Simulation")
    parser.add_argument("--model", type=str, help="Model name to use for the LLMMachine")
    parser.add_argument("--weeks", type=int, help="Number of weeks to simulate")
    parser.add_argument("--json-output", action="store_true", help="Output results as JSON to stdout")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (default is True usually, but disabled if json-output is on)")
    
    args = parser.parse_args()

    # If asking for JSON output, we want to suppress standard prints unless explicitly verbose
    # But existing code prints a lot. We can control verbosity via the `verbose` arg of run_competition.
    # If json-output is True, default verbose to False.
    verbose = args.verbose
    if not args.json_output and not args.verbose:
        # If running normally without flags, default to verbose=True as before
        verbose = True

    if verbose:
        print("=== Script 2: Competitive LLM Vending Machine Simulation ===\n")

    result = run_competition(
        model_name=args.model, 
        verbose=verbose, 
        num_weeks=args.weeks,
        # We can default save_plot and record_history to True for standalone usage,
        # but the caller (benchmark) might probably ignore the plot path anyway or we can add flags for them later if needed.
        # For now, keep them True as in original main.
        save_plot=True,
        record_history=True
    )

    if args.json_output:
        # Print ONLY the JSON result to stdout for parsing
        print(json.dumps(result))

if __name__ == "__main__":
    main()
