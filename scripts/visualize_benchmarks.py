import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

INPUT_FILE = "/Users/andressantos/Desktop/Code/LLM_Vending_Machine/data/benchmarks/benchmark_combined_20251225.csv"

def plot_metric(df, metric_basic, metric_llm, ylabel, title, output_file):
    records = []
    for idx, row in df.iterrows():
        # 1. Basic Machine Entry
        records.append({
            "week": row["week"],
            "value": row[metric_basic],
            "model_label": "Basic Machine",
            "type": "No-Agent",
            "run_index": row["run_index"]
        })
        
        # 2. LLM Model Entry
        records.append({
            "week": row["week"],
            "value": row[metric_llm],
            "model_label": row["model"],
            "type": row["agent_type"],
            "run_index": row["run_index"]
        })

    plot_df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=plot_df,
        x="week",
        y="value",
        hue="model_label",
        style="type",
        markers=True,
        dashes=True,
        errorbar=('ci', 95),
        linewidth=2.5,
        markersize=8
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Week", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    sns.move_legend(
        plt.gca(), "lower center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=3,
        title="Model & Type",
        frameon=True,
    )
    plt.subplots_adjust(bottom=0.3)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close() # Close figure to free memory

def visualize_benchmarks():
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Plot Profit
    plot_metric(
        df, 
        "basic_profit", 
        "llm_profit", 
        "Profit", 
        "Benchmarking: Profit over Weeks (Agent vs. No-Agent)", 
        "/Users/andressantos/Desktop/Code/LLM_Vending_Machine/data/benchmarks/benchmark_profit_comparison_20251225.png"
    )
    
    # Plot Avg Price
    plot_metric(
        df, 
        "basic_avg_price", 
        "llm_avg_price", 
        "Average Price", 
        "Benchmarking: Average Price over Weeks (Agent vs. No-Agent)", 
        "/Users/andressantos/Desktop/Code/LLM_Vending_Machine/data/benchmarks/benchmark_price_comparison_20251225.png"
    )

if __name__ == "__main__":
    visualize_benchmarks()
