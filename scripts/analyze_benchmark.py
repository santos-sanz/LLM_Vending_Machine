"""
Aggregate a benchmark_raw_*.csv into a statistical summary with dispersion
(mean, stdev, min, max, median, 95% CI) and a Markdown analysis report.

Uses only the Python standard library (no pandas) for portability.

Usage:
    python3 scripts/analyze_benchmark.py --raw data/benchmarks/benchmark_raw_<ts>.csv
    python3 scripts/analyze_benchmark.py --raw <csv> --weeks 52 --seed-base 42 --runs 5
"""
import os
import sys
import csv
import argparse
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import model_type

# Two-sided t critical values for 95% CI with n-1 dof (small samples)
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ci95(values):
    n = len(values)
    if n < 2:
        return 0.0
    t = T_CRITICAL_95.get(n - 1, 1.96)
    return t * statistics.stdev(values) / (n ** 0.5)


def load_runs(raw_path):
    """Group weekly rows into runs: (model, run_index) -> list of week rows."""
    runs = defaultdict(list)
    with open(raw_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model"], int(row["run_index"]))
            runs[key].append({
                "week": int(row["week"]),
                "seed": int(row["seed"]),
                "basic_profit": float(row["basic_profit"]),
                "llm_profit": float(row["llm_profit"]),
                "basic_avg_price": float(row["basic_avg_price"]),
                "llm_avg_price": float(row["llm_avg_price"]),
                "basic_avg_stock": float(row["basic_avg_stock"]),
                "llm_avg_stock": float(row["llm_avg_stock"]),
            })
    for key in runs:
        runs[key].sort(key=lambda r: r["week"])
    return runs


def summarize_model(model, run_finals, run_margins, run_prices, seeds):
    def stats_block(values, ndigits=2):
        vals = [round(v, ndigits) for v in values]
        return {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "ci95": ci95(vals),
            "min": min(vals),
            "max": max(vals),
        }

    llm = stats_block(run_finals)
    margin = stats_block(run_margins)
    prices = stats_block(run_prices)
    wins = sum(1 for m in run_margins if m > 0)
    return {
        "model": model,
        "model_type": model_type(model),
        "seeds": sorted(set(seeds)),
        "llm": llm,
        "margin": margin,
        "prices": prices,
        "win_rate": 100.0 * wins / len(run_margins),
        "wins": wins,
        "n_runs": len(run_finals),
    }


def fmt(v, nd=0):
    return f"{v:,.{nd}f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark raw results with dispersion statistics")
    parser.add_argument("--raw", type=str, required=True, help="Path to benchmark_raw_*.csv")
    parser.add_argument("--runs", type=int, default=5, help="Expected runs per model (for the methodology note)")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed used (for the methodology note)")
    parser.add_argument("--weeks", type=int, default=52, help="Weeks per run (for the methodology note)")
    parser.add_argument("--out", type=str, default=None, help="Output markdown path (default: alongside raw csv)")
    args = parser.parse_args()

    runs = load_runs(args.raw)
    by_model = defaultdict(lambda: {"finals": [], "margins": [], "prices": [], "seeds": [], "per_run": []})

    for (model, run_idx), weeks in sorted(runs.items()):
        final = weeks[-1]
        by_model[model]["finals"].append(final["llm_profit"])
        by_model[model]["margins"].append(final["llm_profit"] - final["basic_profit"])
        by_model[model]["prices"].append(final["llm_avg_price"])
        by_model[model]["seeds"].append(final["seed"])
        by_model[model]["per_run"].append((run_idx, final))

    summaries = []
    for model, data in by_model.items():
        summaries.append(summarize_model(model, data["finals"], data["margins"], data["prices"], data["seeds"]))
    summaries.sort(key=lambda s: s["llm"]["mean"], reverse=True)

    # --- Console table ---
    print("=" * 116)
    print(f"{'AGGREGATED BENCHMARK (dispersion across runs)':^116}")
    print("=" * 116)
    print(f"{'Model':<26} | {'Type':<5} | {'N':<2} | {'Mean':>9} | {'±σ':>7} | {'95% CI':>15} | {'Median':>9} | {'Min':>8} | {'Max':>8} | {'Win%':>5}")
    print("-" * 116)
    for s in summaries:
        lo, hi = s["llm"]["mean"] - s["llm"]["ci95"], s["llm"]["mean"] + s["llm"]["ci95"]
        print(f"{s['model']:<26} | {s['model_type']:<5} | {s['n_runs']:<2} | {fmt(s['llm']['mean']):>9} | {fmt(s['llm']['stdev']):>7} | "
              f"[{fmt(lo)}, {fmt(hi)}] | {fmt(s['llm']['median']):>9} | {fmt(s['llm']['min']):>8} | {fmt(s['llm']['max']):>8} | {s['win_rate']:>4.0f}%")
    print("-" * 116)
    print(f"{'Model':<26} | {'Margin mean':>11} | {'Margin ±σ':>9} | {'Margin min-max':>17} | {'Avg final price':>15} | {'Wins':>5}")
    for s in summaries:
        print(f"{s['model']:<26} | {fmt(s['margin']['mean']):>11} | {fmt(s['margin']['stdev']):>9} | "
              f"[{fmt(s['margin']['min'])}, {fmt(s['margin']['max'])}] | {fmt(s['prices']['mean'], 2):>15} | {s['wins']:>2}/{s['n_runs']}")
    print("=" * 116)

    # --- Markdown report ---
    best = summaries[0]
    lines = []
    lines.append("# LLM Vending Machine Benchmark — Aggregated Analysis (nan + OpenRouter decision models)")
    lines.append("")
    lines.append(f"- Source: `{args.raw}`")
    lines.append(f"- Simulation: agentic (LangGraph) mode, {args.weeks} weeks per run, LLMMachine vs static BasicMachine")
    lines.append(f"- Reproducibility: {args.runs} runs per model with paired seeds `{args.seed_base}..{args.seed_base + args.runs - 1}` (every model faces the same seeded demand sequences)")
    lines.append(f"- Providers: nan API (chat models via chat/completions) and OpenRouter (the top-10 weekly "
                 f"`decisions` models via POST /api/alpha/decisions: one focused `choice` question per product "
                 f"over a symmetric ±7.5%/±15% price-action menu, with a flat product-scoped state)")
    lines.append(f"- Exclusions: mimo-v2.6-flash (nan allowance exhausted), minimax-h3 (no key access), "
                 f"gemma4 (reasoning chains overflow the gateway timeout and its 5-request concurrency cap "
                 f"makes full runs unreliable)")
    lines.append("")
    lines.append("## Final net profit per model (aggregate over runs)")
    lines.append("")
    lines.append("| Model | Type | N | Mean | ±σ | 95% CI | Median | Min | Max | Win rate |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    for s in summaries:
        lo, hi = s["llm"]["mean"] - s["llm"]["ci95"], s["llm"]["mean"] + s["llm"]["ci95"]
        lines.append(f"| `{s['model']}` | {s['model_type']} | {s['n_runs']} | {fmt(s['llm']['mean'], 1)} | {fmt(s['llm']['stdev'], 1)} "
                     f"| [{fmt(lo)}, {fmt(hi)}] | {fmt(s['llm']['median'], 1)} | {fmt(s['llm']['min'], 1)} "
                     f"| {fmt(s['llm']['max'], 1)} | {s['win_rate']:.0f}% ({s['wins']}/{s['n_runs']}) |")
    lines.append("")
    lines.append("## Head-to-head margin vs BasicMachine (LLM profit − Basic profit, final week)")
    lines.append("")
    lines.append("| Model | Type | Margin mean | ±σ | Min | Max | Wins |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(f"| `{s['model']}` | {s['model_type']} | {fmt(s['margin']['mean'], 1)} | {fmt(s['margin']['stdev'], 1)} "
                     f"| {fmt(s['margin']['min'], 1)} | {fmt(s['margin']['max'], 1)} | {s['wins']}/{s['n_runs']} |")
    lines.append("")
    lines.append("## Per-run detail (final week)")
    lines.append("")
    lines.append("| Model | Run | Seed | Basic profit | LLM profit | Margin | LLM avg price |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        for run_idx, final in sorted(by_model[s["model"]]["per_run"]):
            lines.append(f"| `{s['model']}` | {run_idx} | {final['seed']} | {fmt(final['basic_profit'], 1)} "
                         f"| {fmt(final['llm_profit'], 1)} | {fmt(final['llm_profit'] - final['basic_profit'], 1)} "
                         f"| {final['llm_avg_price']:.2f} |")
    lines.append("")
    lines.append("## Notes on dispersion")
    lines.append("")
    cv = {s["model"]: (s["llm"]["stdev"] / abs(s["llm"]["mean"]) * 100 if s["llm"]["mean"] else float("nan")) for s in summaries}
    lines.append("- Coefficient of variation (σ/mean) of final profit: "
                 + ", ".join(f"`{m}` {v:.1f}%" for m, v in cv.items()))
    lines.append("- Dispersion has two sources: (1) LLM sampling stochasticity (temperature 0.7, reasoning models) "
                 "and (2) market randomness, which is seeded identically per run index across models (paired seeds), "
                 "so differences between models are not explained by luckier demand draws.")
    lines.append(f"- Best mean performer: `{best['model']}` with {fmt(best['llm']['mean'], 1)} "
                 f"(95% CI ± {fmt(best['llm']['ci95'], 1)}), winning {best['wins']}/{best['n_runs']} runs against BasicMachine.")
    lines.append("")

    out_path = args.out or os.path.join(
        os.path.dirname(args.raw),
        "benchmark_analysis_" + os.path.splitext(os.path.basename(args.raw))[0].replace("benchmark_raw_", "") + ".md"
    )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report saved: {out_path}")


if __name__ == "__main__":
    main()
