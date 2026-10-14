# Final Ranking — LLM Vending Machine Benchmark (2026-10-14)

**Champion: `cloudflare/clef`** (decider) — mean net profit **2,451 ± 124** over 5 runs of 52 weeks, winning 100% of runs against the static BasicMachine (strategy: aggressive price-maker (5x+ baseline), avg final price 10.23 vs baseline 1.62).

Best by type: **cloudflare/clef** (decider, 2,451) vs **glm5.3-flash** (standard, 1,289).

## Ranking

| # | Model | Type | Mean profit | ±σ | 95% CI | Median | Min–Max | Win rate | Margin vs Basic | Avg final price | Strategy |
|---:|---|---|---:|---:|---|---:|---|---:|---:|---:|---|
| 1 | `cloudflare/clef` | decider | 2,450.7 | 123.5 | [2,297, 2,604] | 2,430.6 | [2,346, 2,654] | 100% (5/5) | 1,964.4 ± 128.3 | 10.23 | Aggressive price-maker (5x+ baseline) |
| 2 | `glm5.3-flash` | standard | 1,289.0 | 409.6 | [781, 1,797] | 1,280.4 | [755, 1,877] | 100% (5/5) | 946.0 ± 375.8 | 4.31 | Gradual price hiker |
| 3 | `deepseek-v4-flash` | standard | 1,134.7 | 435.8 | [594, 1,676] | 1,167.3 | [554, 1,749] | 100% (5/5) | 804.8 ± 430.6 | 3.55 | Gradual price hiker |
| 4 | `qwen3.8-flash` | standard | 728.3 | 267.2 | [397, 1,060] | 737.8 | [312, 1,034] | 80% (4/5) | 413.2 ± 276.4 | 2.48 | Gradual price hiker |
| 5 | `openai/gpt-6-luna-decisions` | decider | 709.7 | 122.4 | [558, 862] | 727.4 | [544, 868] | 60% (3/5) | -5.1 ± 122.0 | 369.83 | Aggressive price-maker (5x+ baseline) |
| 6 | `qwen3.6` | standard | 443.0 | 35.9 | [399, 488] | 427.5 | [408, 491] | 100% (5/5) | 139.7 ± 54.4 | 1.94 | Cautious / static pricing |
| 7 | `nace-ai/drex-v1.5` | decider | 399.4 | 92.5 | [285, 514] | 415.9 | [281, 493] | 80% (4/5) | 83.9 ± 83.1 | 2.60 | Gradual price hiker |
| 8 | `liquid/d1` | decider | 277.0 | 33.4 | [236, 318] | 282.2 | [226, 310] | 20% (1/5) | -20.1 ± 36.8 | 1.62 | Cautious / static pricing |
| 9 | `inception/mercury-decide:free` | decider | 261.5 | 35.4 | [218, 305] | 257.6 | [209, 296] | 20% (1/5) | -34.9 ± 36.5 | 1.57 | Cautious / static pricing |
| 10 | `typesafe/jev-1.13` | decider | 261.3 | 36.3 | [216, 306] | 278.8 | [205, 294] | 20% (1/5) | -34.6 ± 40.9 | 1.57 | Cautious / static pricing |
| 11 | `cloudflare/clef-flash` | decider | 207.8 | 34.0 | [166, 250] | 217.9 | [155, 241] | 0% (0/5) | -84.9 ± 39.6 | 1.43 | Under-cutter (below baseline) |
| 12 | `perplexity/pplx-decider-v1.1-27b` | decider | 22.1 | 20.5 | [-3, 48] | 35.0 | [-2, 39] | 0% (0/5) | -261.6 ± 33.0 | 1.09 | Under-cutter (below baseline) |
| 13 | `microsoft/microsoft-decision-1` | decider | -140.5 | 71.1 | [-229, -52] | -101.6 | [-226, -78] | 0% (0/5) | -412.2 ± 83.9 | 0.71 | Under-cutter (below baseline) |
| 14 | `upstage/solar-decide` | decider | -169.6 | 25.7 | [-201, -138] | -160.6 | [-200, -141] | 0% (0/5) | -447.8 ± 25.2 | 0.72 | Under-cutter (below baseline) |

## Key insights

- **Pricing up beats volume.** Every model that finished in the black with a big margin systematically raised prices; the static baseline (BasicMachine) is beatable because demand is price-inelastic enough (low price sensitivity) to absorb significant hikes.
- **Decision models expose real strategic diversity** once given a properly formatted, product-scoped state: from `cloudflare/clef`'s disciplined price escalations to the death-spiral undercutting of `upstage/solar-decide` and `microsoft/microsoft-decision-1` (final prices below unit cost → guaranteed losses).
- **Dispersion matters**: `glm5.3-flash` and `deepseek-v4-flash` reach high means but with σ ≈ 400–440, i.e. inconsistent strategies; the top decision models win with tighter dispersion (clef σ ≈ 124).
- **Cautious agents lose by drift**: models that mostly `keep` prices track BasicMachine's profit minus a small random-split drag (same seeds → near-identical trajectories).

## Methodology

- Source: `data/benchmarks/benchmark_combined_20261014_full.csv` — 14 models × 5 runs × 52 weeks, agentic (LangGraph) mode, LLMMachine vs static BasicMachine sharing one client pool.
- **Paired seeds 42–46**: every model faces the same seeded demand sequences, so ranking differences reflect strategy, not luckier demand draws.
- **Standard models** (nan API): free-form JSON tool calling over chat/completions; per-model overrides bound hidden reasoning of thinking models.
- **Decider models** (OpenRouter top-10 weekly `decisions`): one focused `choice` question per product per week over a symmetric price-action menu (keep / ±7.5% / ±15%), flat product-scoped state via `POST /api/alpha/decisions`.
- Dispersion across runs covers LLM sampling stochasticity (temperature 0.7) and residual market randomness; 95% CI uses Student-t with n−1 dof.
- Exclusions: mimo-v2.6-flash (nan allowance exhausted), minimax-h3 (no key access), gemma4 (gateway timeouts + 5-req concurrency cap).
