import pandas as pd
import os

# Define file paths
FILE_NO_AGENT = "/Users/andressantos/Desktop/Code/LLM_Vending_Machine/data/benchmarks/benchmark_raw_20251225_020208.csv"
FILE_AGENT = "/Users/andressantos/Desktop/Code/LLM_Vending_Machine/data/benchmarks/benchmark_raw_20251225_185128.csv"
OUTPUT_FILE = "/Users/andressantos/Desktop/Code/LLM_Vending_Machine/data/benchmarks/benchmark_combined_20251225.csv"

def merge_benchmarks():
    print("Loading benchmark files...")
    
    # Load separate dataframes
    if not os.path.exists(FILE_NO_AGENT):
        print(f"Error: File not found: {FILE_NO_AGENT}")
        return
    if not os.path.exists(FILE_AGENT):
        print(f"Error: File not found: {FILE_AGENT}")
        return

    df_no_agent = pd.read_csv(FILE_NO_AGENT)
    df_agent = pd.read_csv(FILE_AGENT)

    # Label them
    df_no_agent['agent_type'] = 'No-Agent'
    df_agent['agent_type'] = 'Agent'

    # Concatenate
    df_combined = pd.concat([df_no_agent, df_agent], ignore_index=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully merged files into {OUTPUT_FILE}")
    print(f"Total rows: {len(df_combined)}")
    print(df_combined.head())

if __name__ == "__main__":
    merge_benchmarks()
