import argparse
from src.hf_competitive_run import run_hf_competition

def main():
    parser = argparse.ArgumentParser(description="Benchmark HF: SmolLM3-3B vs BasicMachine")
    parser.add_argument("--weeks", type=int, default=12, help="Number of weeks to simulate")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Model name to benchmark")
    args = parser.parse_args()

    print(f"--- Running Benchmark HF: {args.model} vs BasicMachine ---")
    run_hf_competition(models=[args.model], num_weeks=args.weeks)

if __name__ == "__main__":
    main()
