import os
import time
import csv
import json
import subprocess
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.config import BENCHMARKS_DIR, NAN_CHAT_MODELS, DEFAULT_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Vending Machine Benchmarking")
    parser.add_argument("--mode", type=str, choices=["agentic", "legacy"], default="agentic", help="Simulation mode")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per model")
    parser.add_argument("--weeks", type=int, default=52, help="Number of weeks per simulation")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models (default: all nan chat models)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Run i uses seed = seed + i (identical seeds across models for a fair paired comparison)")
    parser.add_argument("--workers", type=int, default=8, help="Number of simulations to run in parallel")
    return parser.parse_args()


def run_single(model, weeks, seed, mode):
    """Run one simulation as a subprocess and return the parsed JSON result."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        "python3", "src/agentic_run.py",
        "--model", model,
        "--weeks", str(weeks),
        "--mode", mode,
        "--json-output",
        "--seed", str(seed),
    ]

    # 90-minute hard timeout per run to prevent infinite hangs
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=5400)

    if proc.returncode != 0:
        raise RuntimeError(f"Process failed with return code {proc.returncode}: {proc.stderr[-1500:]}")

    output = proc.stdout.strip()
    lines = [l for l in output.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"No output produced. Stderr: {proc.stderr[-1500:]}")

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        # Fallback: try to parse the whole output as JSON
        return json.loads(output)


def aggregate_model_results(model, results):
    """Compute aggregated stats (including dispersion) for one model."""
    llm_profits = [float(r["llm_profit"]) for r in results]
    basic_profits = [float(r["basic_profit"]) for r in results]
    margins = [l - b for l, b in zip(llm_profits, basic_profits)]
    wins = sum(1 for m in margins if m > 0)

    def stats(values):
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    llm_stats = stats(llm_profits)
    basic_stats = stats(basic_profits)
    margin_stats = stats(margins)

    return {
        "model": model,
        "total_runs": len(results),
        "llm_profit_mean": round(llm_stats["mean"], 2),
        "llm_profit_stdev": round(llm_stats["stdev"], 2),
        "llm_profit_min": round(llm_stats["min"], 2),
        "llm_profit_max": round(llm_stats["max"], 2),
        "llm_profit_median": round(llm_stats["median"], 2),
        "basic_profit_mean": round(basic_stats["mean"], 2),
        "basic_profit_stdev": round(basic_stats["stdev"], 2),
        "margin_mean": round(margin_stats["mean"], 2),
        "margin_stdev": round(margin_stats["stdev"], 2),
        "margin_min": round(margin_stats["min"], 2),
        "margin_max": round(margin_stats["max"], 2),
        "win_rate": round((wins / len(results)) * 100, 1),
    }


def print_summary(sorted_summary):
    print("\n" + "=" * 115)
    print(f"{'Benchmarking Summary (aggregated over runs)':^115}")
    print("=" * 115)
    header = (f"{'Model':<28} | {'Runs':<4} | {'LLM mean':<10} | {'LLM ±σ':<10} | "
              f"{'LLM min-max':<17} | {'Basic mean':<10} | {'Margin ±σ':<10} | {'Win %':<6}")
    print(header)
    print("-" * 115)
    for res in sorted_summary:
        minmax = f"[{res['llm_profit_min']:.0f}, {res['llm_profit_max']:.0f}]"
        print(f"{res['model']:<28} | {res['total_runs']:<4} | {res['llm_profit_mean']:<10.2f} | "
              f"{res['llm_profit_stdev']:<10.2f} | {minmax:<17} | {res['basic_profit_mean']:<10.2f} | "
              f"{res['margin_stdev']:<10.2f} | {res['win_rate']:<6.1f}%")
    print("=" * 115)


def main():
    args = parse_args()

    models = args.models.split(",") if args.models else list(NAN_CHAT_MODELS)
    # Treat "default" as a placeholder for the configured default model
    models = [DEFAULT_MODEL if m.strip() == "default" else m.strip() for m in models]

    total = len(models) * args.runs
    print(f"=== LLM Vending Machine Benchmarking Pipeline ({args.mode.upper()} Mode) ===\n")
    print("Benchmark Configuration:")
    print(f"- Models: {len(models)} -> {models}")
    print(f"- Runs per model: {args.runs}")
    print(f"- Weeks per run: {args.weeks}")
    print(f"- Seeds: {args.seed}..{args.seed + args.runs - 1} (same seeds across models)")
    print(f"- Parallel workers: {args.workers}")
    print(f"- Total simulations: {total}\n")

    # (model, run_index) -> seed; identical seeds across models = paired comparison
    jobs = [(model, i, args.seed + i) for model in models for i in range(1, args.runs + 1)]

    results_by_model = {m: [] for m in models}
    raw_rows = []
    failures = []
    done_count = 0

    def execute_jobs(job_list):
        nonlocal done_count
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_single, model, args.weeks, seed, args.mode): (model, i, seed)
                for model, i, seed in job_list
            }
            for future in as_completed(futures):
                model, i, seed = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    results_by_model[model].append(result)
                    print(f"[{done_count}/{total}] OK {model} run={i} seed={seed} "
                          f"basic={float(result['basic_profit']):.2f} llm={float(result['llm_profit']):.2f}", flush=True)

                    for week_data in result.get("weekly_stats", []):
                        raw_rows.append({
                            "timestamp": result.get("timestamp"),
                            "model": model,
                            "run_index": i,
                            "seed": seed,
                            "week": week_data["week"],
                            "basic_profit": week_data["basic_profit"],
                            "llm_profit": week_data["llm_profit"],
                            "basic_avg_price": week_data["basic_avg_price"],
                            "llm_avg_price": week_data["llm_avg_price"],
                            "basic_avg_stock": week_data["basic_avg_stock"],
                            "llm_avg_stock": week_data["llm_avg_stock"]
                        })
                except subprocess.TimeoutExpired:
                    failures.append((model, i, seed, "timeout"))
                    print(f"[{done_count}/{total}] TIMEOUT {model} run={i} seed={seed}", flush=True)
                except Exception as e:
                    failures.append((model, i, seed, str(e)[:200]))
                    print(f"[{done_count}/{total}] ERROR {model} run={i} seed={seed}: {str(e)[:200]}", flush=True)

    execute_jobs(jobs)

    # Retry failed jobs once (transient API errors / gateway timeouts)
    if failures:
        retriable = [(m, i, s) for m, i, s, _ in failures]
        failures.clear()
        print(f"\n[Retry] Retrying {len(retriable)} failed run(s)...")
        done_count = 0
        total = len(retriable)
        execute_jobs(retriable)

    if failures:
        print(f"\n[Warning] {len(failures)} run(s) failed:")
        for model, i, seed, err in failures:
            print(f"  - {model} run={i} seed={seed}: {err}")

    all_summary_results = [
        aggregate_model_results(model, results)
        for model, results in results_by_model.items() if results
    ]
    sorted_summary = sorted(all_summary_results, key=lambda x: x["llm_profit_mean"], reverse=True)
    print_summary(sorted_summary)

    # Persistence
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(BENCHMARKS_DIR, f"benchmark_raw_{timestamp}.csv")
    with open(raw_path, mode="w", newline="") as f:
        if raw_rows:
            writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
            writer.writeheader()
            writer.writerows(raw_rows)

    summary_path = os.path.join(BENCHMARKS_DIR, f"benchmark_summary_{timestamp}.csv")
    with open(summary_path, mode="w", newline="") as f:
        if sorted_summary:
            writer = csv.DictWriter(f, fieldnames=sorted_summary[0].keys())
            writer.writeheader()
            writer.writerows(sorted_summary)

    print(f"\nBenchmark completed.")
    print(f"- Raw results saved: {raw_path}")
    print(f"- Summary report saved: {summary_path}")


if __name__ == "__main__":
    main()
