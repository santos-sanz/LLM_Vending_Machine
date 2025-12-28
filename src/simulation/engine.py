import os
import random
import numpy as np
from src.models.vending_machine import VendingMachine
from src.config import CLIENT_DISTRIBUTION, CLIENT_LAMBDA, CLIENT_UNIFORM_RANGE

def simulate_week(vending_machine: VendingMachine) -> float:
    if not os.getenv("RL_TRAINING"):
        print("\n--- Simulating One Week of Operation ---")

    # 1. Recharging products
    vending_machine.recharge_products()

    # 2. Applying maintenance costs
    vending_machine.apply_maintenance()

    # 3. Simulate client purchase attempts
    if CLIENT_DISTRIBUTION == "poisson":
        clients = np.random.poisson(CLIENT_LAMBDA)
    else:
        clients = random.randint(*CLIENT_UNIFORM_RANGE)
    if not os.getenv("RL_TRAINING"):
        print(f"Simulating {clients} purchase attempts...")
    for _ in range(clients):
        # Get product names and their likelihoods
        product_names = [p.name for p in vending_machine.products.values()]
        purchase_likelihoods = [p.purchase_likelihood for p in vending_machine.products.values()]

        # Normalize likelihoods to act as weights for random.choices
        total_likelihood = sum(purchase_likelihoods)
        if total_likelihood == 0:
            # If no products have a likelihood, skip purchases
            print("No products available for purchase.")
            break
        normalized_likelihoods = [l / total_likelihood for l in purchase_likelihoods]

        # Randomly select a product based on its purchase likelihood
        selected_product_name = random.choices(product_names, weights=normalized_likelihoods, k=1)[0]

        # Attempt to sell the product
        vending_machine.sell_product(selected_product_name)

    # 4. Calculate the net profit for the week
    weekly_profit_loss = vending_machine.calculate_profit_loss()
    if not os.getenv("RL_TRAINING"):
        print(f"End of week cash: {vending_machine.cash:.2f}")
        print(f"Weekly Net Profit/Loss (cash - initial investment): {weekly_profit_loss:.2f}")

    return weekly_profit_loss

def simulate_competitive_week(vending_machines: list[VendingMachine], week_number: int) -> tuple[list[float], list[str]]:
    if not os.getenv("RL_TRAINING"):
        print(f"\n--- Simulating Competitive Week {week_number} of Operation ---")
    stockout_events = []
    reported_stockouts = set() # To only report the first time a product runs out in a week

    for vm in vending_machines:
        # 1. Recharging products
        vm.recharge_products()
        # 2. Applying maintenance costs
        vm.apply_maintenance()

    # 3. Simulate client purchase attempts
    if CLIENT_DISTRIBUTION == "poisson":
        clients = np.random.poisson(CLIENT_LAMBDA)
    else:
        clients = random.randint(*CLIENT_UNIFORM_RANGE)
    if not os.getenv("RL_TRAINING"):
        print(f"Simulating {clients} purchase attempts across {len(vending_machines)} machines...")
    
    for client_idx in range(1, clients + 1):
        # Gather all available products across all machines
        all_options = [] # list of (vm, product_name, likelihood)
        
        # Check for stockouts to report
        for i, vm in enumerate(vending_machines):
            for p_name, p in vm.products.items():
                if p.stock == 0:
                    event_key = (i, p_name)
                    if event_key not in reported_stockouts:
                        msg = f"[Week {week_number}, Client {client_idx}] Machine {i}: {p_name} is out of stock!"
                        print(msg)
                        stockout_events.append(msg)
                        reported_stockouts.add(event_key)

        # Fix: Gather all available products properly
        all_options = []
        for i, vm in enumerate(vending_machines):
            for p in vm.products.values():
                if p.stock > 0:
                    all_options.append((vm, p.name, p.purchase_likelihood))
        
        if not all_options:
            # Only report final stockout if not already reported
            break
            
        # Clients choose based on likelihood (which factors in price)
        vms_list, names, likelihoods = zip(*all_options)
        total_l = sum(likelihoods)
        if total_l == 0:
            continue
            
        weights = [l/total_l for l in likelihoods]
        chosen_idx = random.choices(range(len(all_options)), weights=weights, k=1)[0]
        chosen_vm, chosen_name, _ = all_options[chosen_idx]
        
        chosen_vm.sell_product(chosen_name)

    # 4. Calculate the net profit for each machine
    results = []
    for i, vm in enumerate(vending_machines):
        profit = vm.calculate_profit_loss()
        results.append(profit)
        if not os.getenv("RL_TRAINING"):
            print(f"Machine {i} End of week cash: {vm.cash:.2f}, Profit: {profit:.2f}")

    return results, stockout_events
