import argparse
from src.hf_competitive_run import run_hf_competition

def main():
    parser = argparse.ArgumentParser(description="Competitive Run HF: Santos-Sanz RL model vs BasicMachine")
    parser.add_argument("--weeks", type=int, default=12, help="Number of weeks to simulate")
    parser.add_argument("--model", type=str, default="santos-sanz/vending-machine-rl-model", help="RL Model name to run")
    args = parser.parse_args()

    print(f"--- Running Competitive Run HF: {args.model} vs BasicMachine ---")
    run_hf_competition(models=[args.model], num_weeks=args.weeks)

if __name__ == "__main__":
    main()
