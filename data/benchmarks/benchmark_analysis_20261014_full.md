# LLM Vending Machine Benchmark — Aggregated Analysis (nan + OpenRouter decision models)

- Source: `data/benchmarks/benchmark_combined_20261014_full.csv`
- Simulation: agentic (LangGraph) mode, 52 weeks per run, LLMMachine vs static BasicMachine
- Reproducibility: 5 runs per model with paired seeds `42..46` (every model faces the same seeded demand sequences)
- Providers: nan API (chat models via chat/completions) and OpenRouter (the top-10 weekly `decisions` models via POST /api/alpha/decisions: one focused `choice` question per product over a symmetric ±7.5%/±15% price-action menu, with a flat product-scoped state)
- Exclusions: mimo-v2.6-flash (nan allowance exhausted), minimax-h3 (no key access), gemma4 (reasoning chains overflow the gateway timeout and its 5-request concurrency cap makes full runs unreliable)

## Final net profit per model (aggregate over runs)

| Model | Type | N | Mean | ±σ | 95% CI | Median | Min | Max | Win rate |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| `cloudflare/clef` | decider | 5 | 2,450.7 | 123.5 | [2,297, 2,604] | 2,430.6 | 2,346.1 | 2,653.9 | 100% (5/5) |
| `glm5.3-flash` | standard | 5 | 1,289.0 | 409.6 | [781, 1,797] | 1,280.4 | 754.7 | 1,876.8 | 100% (5/5) |
| `deepseek-v4-flash` | standard | 5 | 1,134.7 | 435.8 | [594, 1,676] | 1,167.3 | 553.5 | 1,749.3 | 100% (5/5) |
| `qwen3.8-flash` | standard | 5 | 728.3 | 267.2 | [397, 1,060] | 737.8 | 312.0 | 1,034.1 | 80% (4/5) |
| `openai/gpt-6-luna-decisions` | decider | 5 | 709.7 | 122.4 | [558, 862] | 727.4 | 544.4 | 867.6 | 60% (3/5) |
| `qwen3.6` | standard | 5 | 443.0 | 35.9 | [399, 488] | 427.5 | 407.8 | 491.0 | 100% (5/5) |
| `nace-ai/drex-v1.5` | decider | 5 | 399.4 | 92.5 | [285, 514] | 415.9 | 281.3 | 493.2 | 80% (4/5) |
| `liquid/d1` | decider | 5 | 277.0 | 33.4 | [236, 318] | 282.2 | 225.7 | 309.9 | 20% (1/5) |
| `inception/mercury-decide:free` | decider | 5 | 261.5 | 35.4 | [218, 305] | 257.6 | 208.7 | 296.3 | 20% (1/5) |
| `typesafe/jev-1.13` | decider | 5 | 261.3 | 36.3 | [216, 306] | 278.8 | 204.8 | 293.6 | 20% (1/5) |
| `cloudflare/clef-flash` | decider | 5 | 207.8 | 34.0 | [166, 250] | 217.9 | 155.4 | 240.5 | 0% (0/5) |
| `perplexity/pplx-decider-v1.1-27b` | decider | 5 | 22.1 | 20.5 | [-3, 48] | 35.0 | -1.9 | 39.3 | 0% (0/5) |
| `microsoft/microsoft-decision-1` | decider | 5 | -140.5 | 71.1 | [-229, -52] | -101.6 | -226.2 | -78.3 | 0% (0/5) |
| `upstage/solar-decide` | decider | 5 | -169.6 | 25.7 | [-201, -138] | -160.6 | -200.4 | -141.4 | 0% (0/5) |

## Head-to-head margin vs BasicMachine (LLM profit − Basic profit, final week)

| Model | Type | Margin mean | ±σ | Min | Max | Wins |
|---|---|---:|---:|---:|---:|---:|
| `cloudflare/clef` | decider | 1,964.4 | 128.3 | 1,846.2 | 2,157.3 | 5/5 |
| `glm5.3-flash` | standard | 946.0 | 375.8 | 457.4 | 1,490.3 | 5/5 |
| `deepseek-v4-flash` | standard | 804.8 | 430.6 | 207.2 | 1,397.2 | 5/5 |
| `qwen3.8-flash` | standard | 413.2 | 276.4 | -24.6 | 724.8 | 4/5 |
| `openai/gpt-6-luna-decisions` | decider | -5.1 | 122.0 | -139.8 | 172.5 | 3/5 |
| `qwen3.6` | standard | 139.7 | 54.4 | 68.2 | 200.0 | 5/5 |
| `nace-ai/drex-v1.5` | decider | 83.9 | 83.1 | -8.5 | 180.7 | 4/5 |
| `liquid/d1` | decider | -20.1 | 36.8 | -69.0 | 22.4 | 1/5 |
| `inception/mercury-decide:free` | decider | -34.9 | 36.5 | -80.5 | 8.7 | 1/5 |
| `typesafe/jev-1.13` | decider | -34.6 | 40.9 | -87.8 | 8.1 | 1/5 |
| `cloudflare/clef-flash` | decider | -84.9 | 39.6 | -135.5 | -44.0 | 0/5 |
| `perplexity/pplx-decider-v1.1-27b` | decider | -261.6 | 33.0 | -317.4 | -234.6 | 0/5 |
| `microsoft/microsoft-decision-1` | decider | -412.2 | 83.9 | -529.6 | -340.5 | 0/5 |
| `upstage/solar-decide` | decider | -447.8 | 25.2 | -472.6 | -410.4 | 0/5 |

## Per-run detail (final week)

| Model | Run | Seed | Basic profit | LLM profit | Margin | LLM avg price |
|---|---:|---:|---:|---:|---:|---:|
| `cloudflare/clef` | 1 | 43 | 513.8 | 2,360.0 | 1,846.2 | 10.45 |
| `cloudflare/clef` | 2 | 44 | 496.6 | 2,653.9 | 2,157.3 | 10.04 |
| `cloudflare/clef` | 3 | 45 | 465.2 | 2,430.6 | 1,965.4 | 10.00 |
| `cloudflare/clef` | 4 | 46 | 459.6 | 2,462.8 | 2,003.2 | 10.15 |
| `cloudflare/clef` | 5 | 47 | 496.2 | 2,346.1 | 1,849.9 | 10.49 |
| `glm5.3-flash` | 1 | 43 | 386.4 | 1,876.8 | 1,490.3 | 5.00 |
| `glm5.3-flash` | 2 | 44 | 380.4 | 1,405.9 | 1,025.5 | 4.25 |
| `glm5.3-flash` | 3 | 45 | 333.2 | 1,127.0 | 793.8 | 4.35 |
| `glm5.3-flash` | 4 | 46 | 297.4 | 754.7 | 457.4 | 4.38 |
| `glm5.3-flash` | 5 | 47 | 317.4 | 1,280.4 | 963.1 | 3.56 |
| `deepseek-v4-flash` | 1 | 43 | 339.6 | 1,244.8 | 905.1 | 3.56 |
| `deepseek-v4-flash` | 2 | 44 | 346.4 | 553.5 | 207.1 | 2.25 |
| `deepseek-v4-flash` | 3 | 45 | 352.1 | 1,749.3 | 1,397.2 | 5.25 |
| `deepseek-v4-flash` | 4 | 46 | 298.2 | 958.4 | 660.2 | 4.00 |
| `deepseek-v4-flash` | 5 | 47 | 312.8 | 1,167.3 | 854.5 | 2.69 |
| `qwen3.8-flash` | 1 | 43 | 326.3 | 861.4 | 535.1 | 3.88 |
| `qwen3.8-flash` | 2 | 44 | 336.6 | 312.0 | -24.6 | 1.56 |
| `qwen3.8-flash` | 3 | 45 | 312.9 | 696.3 | 383.5 | 1.79 |
| `qwen3.8-flash` | 4 | 46 | 290.6 | 737.8 | 447.2 | 1.90 |
| `qwen3.8-flash` | 5 | 47 | 309.4 | 1,034.1 | 724.7 | 3.25 |
| `openai/gpt-6-luna-decisions` | 1 | 43 | 741.4 | 643.9 | -97.5 | 487.36 |
| `openai/gpt-6-luna-decisions` | 2 | 44 | 740.5 | 765.1 | 24.5 | 348.18 |
| `openai/gpt-6-luna-decisions` | 3 | 45 | 695.0 | 867.6 | 172.5 | 321.18 |
| `openai/gpt-6-luna-decisions` | 4 | 46 | 684.2 | 544.4 | -139.9 | 351.38 |
| `openai/gpt-6-luna-decisions` | 5 | 47 | 712.6 | 727.4 | 14.7 | 341.04 |
| `qwen3.6` | 1 | 43 | 308.1 | 427.5 | 119.3 | 1.71 |
| `qwen3.6` | 2 | 44 | 339.6 | 407.8 | 68.2 | 1.94 |
| `qwen3.6` | 3 | 45 | 295.9 | 418.5 | 122.6 | 1.62 |
| `qwen3.6` | 4 | 46 | 282.2 | 470.5 | 188.3 | 2.50 |
| `qwen3.6` | 5 | 47 | 291.0 | 491.0 | 200.0 | 1.95 |
| `nace-ai/drex-v1.5` | 1 | 43 | 331.7 | 477.9 | 146.2 | 3.26 |
| `nace-ai/drex-v1.5` | 2 | 44 | 337.0 | 328.5 | -8.5 | 1.70 |
| `nace-ai/drex-v1.5` | 3 | 45 | 322.8 | 415.9 | 93.1 | 3.19 |
| `nace-ai/drex-v1.5` | 4 | 46 | 273.5 | 281.3 | 7.8 | 1.72 |
| `nace-ai/drex-v1.5` | 5 | 47 | 312.5 | 493.2 | 180.7 | 3.13 |
| `liquid/d1` | 1 | 43 | 303.0 | 301.4 | -1.6 | 1.63 |
| `liquid/d1` | 2 | 44 | 334.7 | 265.7 | -69.0 | 1.63 |
| `liquid/d1` | 3 | 45 | 288.4 | 282.2 | -6.2 | 1.62 |
| `liquid/d1` | 4 | 46 | 272.1 | 225.7 | -46.3 | 1.62 |
| `liquid/d1` | 5 | 47 | 287.6 | 309.9 | 22.3 | 1.62 |
| `inception/mercury-decide:free` | 1 | 43 | 302.3 | 291.9 | -10.4 | 1.58 |
| `inception/mercury-decide:free` | 2 | 44 | 333.8 | 253.2 | -80.5 | 1.59 |
| `inception/mercury-decide:free` | 3 | 45 | 288.0 | 257.6 | -30.4 | 1.56 |
| `inception/mercury-decide:free` | 4 | 46 | 270.6 | 208.7 | -62.0 | 1.55 |
| `inception/mercury-decide:free` | 5 | 47 | 287.6 | 296.3 | 8.7 | 1.59 |
| `typesafe/jev-1.13` | 1 | 43 | 301.4 | 283.4 | -18.0 | 1.53 |
| `typesafe/jev-1.13` | 2 | 44 | 333.9 | 246.1 | -87.8 | 1.53 |
| `typesafe/jev-1.13` | 3 | 45 | 286.9 | 278.8 | -8.1 | 1.61 |
| `typesafe/jev-1.13` | 4 | 46 | 271.8 | 204.8 | -67.0 | 1.55 |
| `typesafe/jev-1.13` | 5 | 47 | 285.5 | 293.6 | 8.1 | 1.62 |
| `cloudflare/clef-flash` | 1 | 43 | 299.0 | 217.9 | -81.1 | 1.46 |
| `cloudflare/clef-flash` | 2 | 44 | 330.1 | 194.6 | -135.5 | 1.47 |
| `cloudflare/clef-flash` | 3 | 45 | 284.6 | 240.5 | -44.0 | 1.46 |
| `cloudflare/clef-flash` | 4 | 46 | 269.1 | 155.4 | -113.7 | 1.40 |
| `cloudflare/clef-flash` | 5 | 47 | 280.9 | 230.8 | -50.1 | 1.38 |
| `perplexity/pplx-decider-v1.1-27b` | 1 | 43 | 293.3 | 39.3 | -254.0 | 1.15 |
| `perplexity/pplx-decider-v1.1-27b` | 2 | 44 | 319.1 | 1.6 | -317.4 | 1.06 |
| `perplexity/pplx-decider-v1.1-27b` | 3 | 45 | 275.4 | 35.0 | -240.4 | 1.03 |
| `perplexity/pplx-decider-v1.1-27b` | 4 | 46 | 259.8 | -1.9 | -261.7 | 1.08 |
| `perplexity/pplx-decider-v1.1-27b` | 5 | 47 | 271.4 | 36.7 | -234.6 | 1.12 |
| `microsoft/microsoft-decision-1` | 1 | 43 | 280.1 | -87.5 | -367.6 | 0.78 |
| `microsoft/microsoft-decision-1` | 2 | 44 | 303.4 | -226.2 | -529.6 | 0.51 |
| `microsoft/microsoft-decision-1` | 3 | 45 | 262.9 | -208.8 | -471.6 | 0.58 |
| `microsoft/microsoft-decision-1` | 4 | 46 | 249.9 | -101.6 | -351.5 | 0.78 |
| `microsoft/microsoft-decision-1` | 5 | 47 | 262.2 | -78.3 | -340.5 | 0.90 |
| `upstage/solar-decide` | 1 | 43 | 287.4 | -152.9 | -440.2 | 0.74 |
| `upstage/solar-decide` | 2 | 44 | 312.0 | -160.6 | -472.6 | 0.71 |
| `upstage/solar-decide` | 3 | 45 | 269.1 | -200.4 | -469.5 | 0.71 |
| `upstage/solar-decide` | 4 | 46 | 253.8 | -192.7 | -446.4 | 0.71 |
| `upstage/solar-decide` | 5 | 47 | 268.9 | -141.4 | -410.4 | 0.75 |

## Notes on dispersion

- Coefficient of variation (σ/mean) of final profit: `cloudflare/clef` 5.0%, `glm5.3-flash` 31.8%, `deepseek-v4-flash` 38.4%, `qwen3.8-flash` 36.7%, `openai/gpt-6-luna-decisions` 17.3%, `qwen3.6` 8.1%, `nace-ai/drex-v1.5` 23.2%, `liquid/d1` 12.1%, `inception/mercury-decide:free` 13.5%, `typesafe/jev-1.13` 13.9%, `cloudflare/clef-flash` 16.4%, `perplexity/pplx-decider-v1.1-27b` 92.4%, `microsoft/microsoft-decision-1` 50.6%, `upstage/solar-decide` 15.1%
- Dispersion has two sources: (1) LLM sampling stochasticity (temperature 0.7, reasoning models) and (2) market randomness, which is seeded identically per run index across models (paired seeds), so differences between models are not explained by luckier demand draws.
- Best mean performer: `cloudflare/clef` with 2,450.7 (95% CI ± 153.4), winning 5/5 runs against BasicMachine.
