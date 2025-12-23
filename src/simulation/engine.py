import random
from src.models.vending_machine import VendingMachine

def simulate_week(vending_machine: VendingMachine) -> float:
    print("\n--- Simulating One Week of Operation ---")

    # 1. Recharging products
    vending_machine.recharge_products()

    # 2. Applying maintenance costs
    vending_machine.apply_maintenance()

    # 3. Simulate client purchase attempts
    clients = random.randint(50, 150)
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
    print(f"End of week cash: {vending_machine.cash:.2f}")
    print(f"Weekly Net Profit/Loss (cash - initial investment): {weekly_profit_loss:.2f}")

    return weekly_profit_loss
