import random
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_week

def main():
    print("=== Script 1: Basic Vending Machine Running ===\n")

    # Initialize a vending machine
    # initial_cash, maintenance_cost, initial_investment
    machine = VendingMachine(initial_cash=500.0, maintenance_cost=50.0, initial_investment=0.0)

    # Add some products
    # name, price, cost, stock, max_stock, base_likelihood, price_sensitivity
    machine.add_product(Product("Cola", 2.5, 0.8, 10, 20, 0.8, 0.1))
    machine.add_product(Product("Water", 1.5, 0.3, 15, 30, 0.9, 0.2))
    machine.add_product(Product("Chips", 2.0, 0.6, 8, 15, 0.7, 0.15))

    print(f"\nInitial Machine State:\n{machine}")

    # Run simulation for 4 weeks
    total_profit = 0
    for week in range(1, 5):
        print(f"\n--- WEEK {week} ---")
        weekly_profit = simulate_week(machine)
        total_profit += weekly_profit
        print(f"Weekly Machine State:\n{machine}")

    print(f"\n=== Simulation Finished after 4 weeks ===")
    print(f"Total Profit/Loss: {machine.calculate_profit_loss():.2f}")

if __name__ == "__main__":
    main()
