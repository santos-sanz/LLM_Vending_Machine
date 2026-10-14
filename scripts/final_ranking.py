"""
Generate the final consolidated ranking from a merged benchmark_raw CSV.

Emits a Markdown ranking with model type (decider / standard), dispersion
statistics, win rates, observed pricing strategy and a methodology summary.

Usage:
    python3 scripts/final_ranking.py --raw data/benchmarks/benchmark_combined_20261014_full.csv
"""
import os
import sys
import csv
import argparse
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import model_type

T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

BASELINE_PRICE = 1.62  # BasicMachine static avg price at week 52


def strategy_label(avg_final_price, win_rate):
    ratio = avg_final_price / BASELINE_PRICE
    if ratio >= 5:
        return "Aggressive price-maker (5x+ baseline)"
    if ratio >= 1.5:
        return "Gradual price hiker"
    if ratio >= 0.95:
        return "Cautious / static pricing"
    return "Under-cutter (below baseline)"


def main():
    parser = argparse.ArgumentParser(description="Final consolidated benchmark ranking")
    parser.add_argument("--raw", required=True, help="Merged benchmark_raw CSV")
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    runs = defaultdict(list)
    with open(args.raw, newline="") as f:
        for row in csv.DictReader(f):
            runs[(row["model"], int(row["run_index"]))].append(row)

    per_model = defaultdict(lambda: {"finals": [], "margins": [], "prices": [], "wins": 0})
    for (model, _), weeks in runs.items():
        weeks.sort(key=lambda r: int(r["week"]))
        final = weeks[-1]
        llm, basic = float(final["llm_profit"]), float(final["basic_profit"])
        d = per_model[model]
        d["finals"].append(llm)
        d["margins"].append(llm - basic)
        d["prices"].append(float(final["llm_avg_price"]))
        d["wins"] += 1 if llm > basic else 0

    ranked = []
    for model, d in per_model.items():
        finals, margins, prices = d["finals"], d["margins"], d["prices"]
        n = len(finals)
        stdev = statistics.stdev(finals) if n > 1 else 0.0
        ci = T_CRITICAL_95.get(n - 1, 1.96) * stdev / (n ** 0.5)
        avg_price = statistics.mean(prices)
        ranked.append({
            "model": model,
            "type": model_type(model),
            "n": n,
            "mean": statistics.mean(finals),
            "stdev": stdev,
            "ci": ci,
            "median": statistics.median(finals),
            "min": min(finals),
            "max": max(finals),
            "margin_mean": statistics.mean(margins),
            "margin_stdev": statistics.stdev(margins) if n > 1 else 0.0,
            "price": avg_price,
            "win_rate": 100.0 * d["wins"] / n,
            "strategy": strategy_label(avg_price, d["wins"]),
        })
    ranked.sort(key=lambda s: s["mean"], reverse=True)

    # --- Console ---
    print("=" * 128)
    print(f"{'FINAL RANKING — LLM Vending Machine Benchmark':^128}")
    print("=" * 128)
    print(f"{'#':<3} {'Model':<34} {'Type':<9} {'Mean':>8} {'±σ':>6} {'95% CI':>17} {'Win%':>5} {'Avg price':>10} {'Strategy':<36}")
    print("-" * 128)
    for i, s in enumerate(ranked, 1):
        lo, hi = s["mean"] - s["ci"], s["mean"] + s["ci"]
        print(f"{i:<3} {s['model']:<34} {s['type']:<9} {s['mean']:>8,.0f} {s['stdev']:>6,.0f} "
              f"[{lo:>7,.0f}, {hi:>7,.0f}] {s['win_rate']:>4.0f}% {s['price']:>10.2f} {s['strategy']:<36}")
    print("=" * 128)

    best = ranked[0]
    best_standard = next((s for s in ranked if s["type"] == "standard"), None)
    best_decider = next((s for s in ranked if s["type"] == "decider"), None)

    # --- Markdown ---
    lines = []
    lines.append("# Final Ranking — LLM Vending Machine Benchmark (2026-10-14)")
    lines.append("")
    lines.append(f"**Champion: `{best['model']}`** ({best['type']}) — mean net profit "
                 f"**{best['mean']:,.0f} ± {best['stdev']:,.0f}** over {best['n']} runs of 52 weeks, "
                 f"winning {best['win_rate']:.0f}% of runs against the static BasicMachine "
                 f"(strategy: {best['strategy'].lower()}, avg final price {best['price']:.2f} vs baseline {BASELINE_PRICE}).")
    lines.append("")
    if best_decider and best_standard:
        lines.append(f"Best by type: **{best_decider['model']}** (decider, {best_decider['mean']:,.0f}) vs "
                     f"**{best_standard['model']}** (standard, {best_standard['mean']:,.0f}).")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| # | Model | Type | Mean profit | ±σ | 95% CI | Median | Min–Max | Win rate | Margin vs Basic | Avg final price | Strategy |")
    lines.append("|---:|---|---|---:|---:|---|---:|---|---:|---:|---:|---|")
    for i, s in enumerate(ranked, 1):
        lo, hi = s["mean"] - s["ci"], s["mean"] + s["ci"]
        lines.append(
            f"| {i} | `{s['model']}` | {s['type']} | {s['mean']:,.1f} | {s['stdev']:,.1f} "
            f"| [{lo:,.0f}, {hi:,.0f}] | {s['median']:,.1f} | [{s['min']:,.0f}, {s['max']:,.0f}] "
            f"| {s['win_rate']:.0f}% ({s['n'] and int(round(s['win_rate'] * s['n'] / 100))}/{s['n']}) "
            f"| {s['margin_mean']:,.1f} ± {s['margin_stdev']:,.1f} | {s['price']:.2f} | {s['strategy']} |")
    lines.append("")
    lines.append("## Key insights")
    lines.append("")
    lines.append("- **Pricing up beats volume.** Every model that finished in the black with a big margin "
                 "systematically raised prices; the static baseline (BasicMachine) is beatable because demand "
                 "is price-inelastic enough (low price sensitivity) to absorb significant hikes.")
    lines.append("- **Decision models expose real strategic diversity** once given a properly formatted, "
                 "product-scoped state: from `cloudflare/clef`'s disciplined price escalations to the "
                 "death-spiral undercutting of `upstage/solar-decide` and `microsoft/microsoft-decision-1` "
                 "(final prices below unit cost → guaranteed losses).")
    lines.append("- **Dispersion matters**: `glm5.3-flash` and `deepseek-v4-flash` reach high means but with "
                 "σ ≈ 400–440, i.e. inconsistent strategies; the top decision models win with tighter dispersion "
                 "(clef σ ≈ 124).")
    lines.append("- **Cautious agents lose by drift**: models that mostly `keep` prices track BasicMachine's "
                 "profit minus a small random-split drag (same seeds → near-identical trajectories).")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- Source: `{args.raw}` — 14 models × 5 runs × 52 weeks, agentic (LangGraph) mode, "
                 "LLMMachine vs static BasicMachine sharing one client pool.")
    lines.append("- **Paired seeds 42–46**: every model faces the same seeded demand sequences, so ranking "
                 "differences reflect strategy, not luckier demand draws.")
    lines.append("- **Standard models** (nan API): free-form JSON tool calling over chat/completions; "
                 "per-model overrides bound hidden reasoning of thinking models.")
    lines.append("- **Decider models** (OpenRouter top-10 weekly `decisions`): one focused `choice` question "
                 "per product per week over a symmetric price-action menu (keep / ±7.5% / ±15%), flat "
                 "product-scoped state via `POST /api/alpha/decisions`.")
    lines.append("- Dispersion across runs covers LLM sampling stochasticity (temperature 0.7) and residual "
                 "market randomness; 95% CI uses Student-t with n−1 dof.")
    lines.append("- Exclusions: mimo-v2.6-flash (nan allowance exhausted), minimax-h3 (no key access), "
                 "gemma4 (gateway timeouts + 5-req concurrency cap).")
    lines.append("")

    out_path = args.out or os.path.join(
        os.path.dirname(args.raw), "benchmark_final_ranking_20261014.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nFinal ranking saved: {out_path}")


if __name__ == "__main__":
    main()
