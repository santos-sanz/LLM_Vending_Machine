import os
import time
import csv
import json
import subprocess
from datetime import datetime
# from src.competitive_run import run_competition
from src.config import BENCHMARKS_DIR

def main():
    print("=== LLM Vending Machine Benchmarking Pipeline ===\n")

    # Configuration for the benchmark
    MODELS_TO_TEST = [
        "xiaomi/mimo-v2-flash:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "mistralai/devstral-2512:free",
        "nex-agi/deepseek-v3.1-nex-n1:free"
    ]
    
    RUNS_PER_MODEL = 5  # Set to a higher number for a real benchmark
    
    all_raw_results = []
    all_summary_results = []

    print(f"Benchmark Configuration:")
    print(f"- Models: {len(MODELS_TO_TEST)}")
    print(f"- Runs per model: {RUNS_PER_MODEL}")
    print(f"- Total simulations: {len(MODELS_TO_TEST) * RUNS_PER_MODEL}\n")

    for model in MODELS_TO_TEST:
        print(f"\n--- Benchmarking Model: {model} ---")
        model_results = []
        
        for i in range(1, RUNS_PER_MODEL + 1):
            print(f"Run {i}/{RUNS_PER_MODEL} for {model}...")
            try:
                # Explicitly notify that we are waiting for the LLM during the benchmark
                print(f"  [Status] Simulating competitive run in subprocess...")
                
                # Command to run competitive_run.py as a subprocess
                # We need to set PYTHONPATH to include current dir so imports work in subprocess
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()

                cmd = [
                    "python3", "src/competitive_run.py",
                    "--model", model,
                    "--weeks", "52",
                    "--json-output"
                ]

                # Run with timeout to prevent infinite hangs (e.g. 60 minutes limit per run)
                # capture_output=True captures stdout/stderr
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

                if proc.returncode != 0:
                    print(f"  [Error] Process failed with return code {proc.returncode}")
                    print(f"  Stderr: {proc.stderr}")
                    continue

                # Parse the JSON output from stdout
                # The script might output other things if verbose, but we suppressed it with default logic
                # However, if any other libraries print to stdout, we might need to be careful.
                # We expect the last line (or the whole output) to be valid JSON if logic holds.
                # Let's try to find the JSON blob.
                output = proc.stdout.strip()
                # Find the last line that looks like JSON or try to parse the whole thing
                try:
                    # In our competitive_run, we print json.dumps(result) at the very end
                    # But there could be other prints if libraries are noisy.
                    # Let's try to parse the last non-empty line
                    lines = output.splitlines()
                    json_line = lines[-1] if lines else ""
                    result = json.loads(json_line)
                except json.JSONDecodeError:
                    # Fallback: try to find start of json
                    print(f"  [Warning] Could not parse JSON from last line. Raw output length: {len(output)}")
                    # Try to parse the whole output if it's just JSON
                    result = json.loads(output)
                
                model_results.append(result)
                print(f"  [Success] Run completed.")
                
                # Collect raw results for this run (one row per week)
                for week_data in result.get("weekly_stats", []):
                    all_raw_results.append({
                        "timestamp": result.get("timestamp"),
                        "model": model,
                        "run_index": i,
                        "week": week_data["week"],
                        "basic_profit": week_data["basic_profit"],
                        "llm_profit": week_data["llm_profit"],
                        "basic_avg_price": week_data["basic_avg_price"],
                        "llm_avg_price": week_data["llm_avg_price"],
                        "basic_avg_stock": week_data["basic_avg_stock"],
                        "llm_avg_stock": week_data["llm_avg_stock"]
                    })
                
                # Small sleep to avoid hitting OpenRouter rate limits too hard
                time.sleep(2)
            except subprocess.TimeoutExpired:
                print(f"  [Error] Run {i} for {model} timed out!")
            except Exception as e:
                print(f"  [Error] Exception during run {i} for {model}: {e}")
        
        if model_results:
            # Aggregate results for this model
            # Note: result values are strings in the JSON from competitive_run (formatted), need casting
            # modifying extraction to handle string or float logic if needed. 
            # Actually competitive_run returns floats in 'basic_profit', 'llm_profit' at top level
            avg_basic = sum(float(r['basic_profit']) for r in model_results) / len(model_results)
            avg_llm = sum(float(r['llm_profit']) for r in model_results) / len(model_results)
            wins = sum(1 for r in model_results if float(r['llm_profit']) > float(r['basic_profit']))
            
            all_summary_results.append({
                "model": model,
                "avg_basic_profit": avg_basic,
                "avg_llm_profit": avg_llm,
                "win_rate": (wins / len(model_results)) * 100,
                "total_runs": len(model_results)
            })

    # Sort summary results by average LLM profit descending
    sorted_summary = sorted(all_summary_results, key=lambda x: x['avg_llm_profit'], reverse=True)

    # Display Summary Table
    print("\n" + "="*80)
    print(f"{'Benchmarking Summary':^80}")
    print("="*80)
    print(f"{'Model Name':<45} | {'Avg Basic':<10} | {'Avg LLM':<10} | {'Win %':<6}")
    print("-" * 80)
    
    for res in sorted_summary:
        print(f"{res['model']:<45} | {res['avg_basic_profit']:<10.2f} | {res['avg_llm_profit']:<10.2f} | {res['win_rate']:<6.1f}%")
    
    print("="*80)

    # Persistence
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save Raw Results
    raw_path = os.path.join(BENCHMARKS_DIR, f"benchmark_raw_{timestamp}.csv")
    with open(raw_path, mode='w', newline='') as f:
        if all_raw_results:
            writer = csv.DictWriter(f, fieldnames=all_raw_results[0].keys())
            writer.writeheader()
            writer.writerows(all_raw_results)

    # 2. Save Summary Results
    summary_path = os.path.join(BENCHMARKS_DIR, f"benchmark_summary_{timestamp}.csv")
    with open(summary_path, mode='w', newline='') as f:
        if sorted_summary:
            writer = csv.DictWriter(f, fieldnames=sorted_summary[0].keys())
            writer.writeheader()
            writer.writerows(sorted_summary)
    
    print(f"\nBenchmark completed.")
    print(f"- Raw results saved: {raw_path}")
    print(f"- Summary report saved: {summary_path}")

if __name__ == "__main__":
    main()
