import matplotlib.pyplot as plt
from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_week
from src.config import PRODUCT_CONFIGS, INITIAL_CASH, MAINTENANCE_COST, NUM_WEEKS

def main():
    # 1. Create an instance of the VendingMachine class
    vending_machine = VendingMachine(
        initial_cash=INITIAL_CASH, 
        maintenance_cost=MAINTENANCE_COST, 
        initial_investment=0.0
    )
    print("Vending Machine initialized.")

    # 2. Create and add Product instances from product_configs
    for config in PRODUCT_CONFIGS:
        product = Product(**config)
        vending_machine.add_product(product)

    print("Products created and added to vending machine.")

    # 3. Calculate the initial profit or loss
    initial_pnl = vending_machine.calculate_profit_loss()
    print(f"Initial Vending Machine State:\n{vending_machine}")
    print(f"Initial Profit/Loss: {initial_pnl:.2f}")

    # 4. Run the simulation
    weekly_profits = []
    print(f"Starting {NUM_WEEKS}-week simulation...")

    for i in range(NUM_WEEKS):
        print(f"\n--- WEEK {i+1} ---")
        weekly_pnl = simulate_week(vending_machine)
        weekly_profits.append(weekly_pnl)

    print("\nSimulation complete.")
    print(f"Weekly Profit/Loss over {NUM_WEEKS} weeks: {weekly_profits}")
    print(f"Total Profit/Loss after {NUM_WEEKS} weeks: {sum(weekly_profits):.2f}")
    print(f"Average Weekly Profit/Loss: {sum(weekly_profits) / NUM_WEEKS:.2f}")

    # 5. Plot the weekly profits
    weeks = range(1, NUM_WEEKS + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(weeks, weekly_profits, marker='o', linestyle='-', color='skyblue')
    plt.title('Weekly Net Profit Over Time')
    plt.xlabel('Week Number')
    plt.ylabel('Net Profit')
    plt.grid(True)
    plt.xticks(range(0, NUM_WEEKS + 1, 5))
    plt.tight_layout()
    
    print("Optimization: In a real environment, plt.show() would open a window. "
          "For this environment, we'll save the plot as an image.")
    from src.config import IMAGES_DIR
    import os
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_path = os.path.join(IMAGES_DIR, 'simulation_results.png')
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")
    # plt.show() # Uncomment if running in a GUI environment

if __name__ == "__main__":
    main()
