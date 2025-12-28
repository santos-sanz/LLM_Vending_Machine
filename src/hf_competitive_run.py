import json
import csv
import os
from datetime import datetime
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.client import create_llm_client
from src.llm.tools import VendingMachineTools
from src.config import (
    COMPETITION_PRODUCT_CONFIGS, 
    MAINTENANCE_COST, 
    NUM_WEEKS, 
    SIMULATION_RESULTS_CSV
)
from src.utils.helpers import plot_multi_profits, extract_json_objects

def run_hf_competition(models=None, verbose=True, num_weeks=None, save_plot=True):
    """
    Runs a competitive simulation between BasicMachine and one or more HF models.
    models: List of model names or a single model name.
    """
    if models is None:
        models = ["santos-sanz/vending-machine-rl-model", "HuggingFaceTB/SmolLM3-3B"]
    elif isinstance(models, str):
        models = [models]
        
    sim_weeks = num_weeks if num_weeks is not None else 12
    
    model_labels = []
    for i, m in enumerate(models):
        label = "RLMachine" if "rl-model" in m else (f"Model_{i+1}")
        if len(models) == 1 and label == "Model_1":
            label = "LLMMachine"
        model_labels.append(label)

    if verbose:
        models_str = " vs ".join(models)
        print(f"--- Starting HF Competition: Basic vs {models_str} ({sim_weeks} weeks) ---\n")

    # 1. Initialize Machines
    basic_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
    for config in COMPETITION_PRODUCT_CONFIGS:
        basic_machine.add_product(Product(**config))

    machines = {"BasicMachine": basic_machine}
    llm_clients = {}
    
    for label, model_name in zip(model_labels, models):
        machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
        for config in COMPETITION_PRODUCT_CONFIGS:
            machine.add_product(Product(**config))
        machines[label] = machine
        llm_clients[label] = create_llm_client(mode="hf", model_name=model_name)

    # Data collection
    weeks_list = []
    profits_history = {name: [] for name in machines.keys()}
    weekly_stats = []

    tools = VendingMachineTools(machines)

    system_prompt_template = """
    You are a Strategic Business Manager for '{machine_name}'. You are in direct competition with: {competitors}.
    All machines share the same client pool. Clients choose the product with the highest 'purchase_likelihood'.

    IMPORTANT RULES:
    1. AT THE START OF EACH WEEK, YOUR MACHINE IS REFILLED TO MAX CAPACITY.
    2. 'BasicMachine' NEVER changes its prices.
    3. Your goal is to maximize NET PROFIT.

    STRATEGIC ADVICE:
    - Your priority is PROFIT, not just sales volume.
    - If you raise prices, you might sell fewer items, but if the margin increase covers the volume loss, your PROFIT goes up.
    
    AVAILABLE TOOLS (Respond only in JSON):
    1. {{"action": "change_price", "parameters": {{"machine_name": "{machine_name}", "product_name": "...", "new_price": ...}}}}
    2. {{"action": "next_week"}} - to proceed.
    
    Provide your reasoning first, then the JSON block.
    """

    # 3. Simulation Loop
    for week in range(1, sim_weeks + 1):
        # 1. Simulate the week
        active_machines = list(machines.values())
        results, stockout_events = simulate_competitive_week(active_machines, week)

        # 2. Log profits
        weeks_list.append(week)
        stats = {"week": week}
        
        profit_line = f"Week {week} Profits -> "
        for name, machine in machines.items():
            p = machine.calculate_profit_loss()
            profits_history[name].append(p)
            stats[f"{name.lower()}_profit"] = f"{p:.2f}"
            profit_line += f"{name}: {p:.2f}, "
        
        weekly_stats.append(stats)
        if verbose:
            print(profit_line.rstrip(", "))

        # 3. LLM Decisions
        if week < sim_weeks:
            market_data = tools.get_market_data()
            event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
            
            for m_name in llm_clients.keys():
                other_machines = [n for n in machines.keys() if n != m_name]
                competitors_str = ", ".join([f"'{n}'" for n in other_machines])
                
                prompt = f"""
                --- WEEK {week} RESULTS ---
                Stockout Events:
                {event_feedback}
                
                Current Market Data:
                {json.dumps(market_data, indent=2)}
                
                Machines are being refilled NOW. What adjustments will you make for Week {week+1}?
                """
                
                sys_prompt = system_prompt_template.format(machine_name=m_name, competitors=competitors_str)
                llm_response = llm_clients[m_name].complete(prompt, system_prompt=sys_prompt)
                
                if verbose:
                    print(f"\n[{m_name}] Response: {llm_response}")

                commands = extract_json_objects(llm_response)
                for command in commands:
                    if command.get("action") == "change_price":
                        params = command.get("parameters", {})
                        if isinstance(params, dict):
                            params["machine_name"] = m_name
                            try:
                                result = tools.change_price(**params)
                                if verbose:
                                    print(f"[{m_name}] Tool Result: {result}")
                            except Exception as e:
                                print(f"[{m_name}] Error: {e}")

    # 4. Results and Plotting
    final_profits = {name: machine.calculate_profit_loss() for name, machine in machines.items()}

    if verbose:
        print("\n=== HF Competition Finished ===")
        for name, p in final_profits.items():
            print(f"{name} Final Profit: {p:.2f}")

    from src.config import IMAGES_DIR, LOGS_DIR
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_slug = "_vs_".join([m.split("/")[-1] for m in models])
    graph_filename = f"hf_comparison_{models_slug}_{timestamp}.png"
    graph_path = os.path.join(IMAGES_DIR, graph_filename)
    
    if save_plot:
        from src.utils.helpers import plot_profits, plot_multi_profits
        if len(models) == 1:
            plot_profits(weeks_list, profits_history["BasicMachine"], profits_history[model_labels[0]], models[0], graph_path)
        else:
            # For more than 2 models, we might need a more general plot_multi_profits, 
            # but for now we follow the existing helper pattern or just plot the first two.
            plot_multi_profits(weeks_list, profits_history["BasicMachine"], 
                               profits_history[model_labels[0]], profits_history[model_labels[1]], 
                               model_labels[0], model_labels[1], graph_path)

    log_filename = f"hf_log_{models_slug}_{timestamp}.csv"
    log_path = os.path.join(LOGS_DIR, log_filename)
    with open(log_path, mode='w', newline='') as f:
        if weekly_stats:
            writer = csv.DictWriter(f, fieldnames=weekly_stats[0].keys())
            writer.writeheader()
            writer.writerows(weekly_stats)
    
    result_dict = {f"{name.lower()}_profit": p for name, p in final_profits.items()}
    result_dict.update({
        "graph_path": graph_path,
        "log_path": log_path
    })
    return result_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=12)
    args = parser.parse_args()
    
    run_hf_competition(num_weeks=args.weeks)
