# LLM Vending Machine Benchmark — Aggregated Analysis (nan + OpenRouter decision models)

- Source: `data/benchmarks/benchmark_combined_20261014.csv`
- Simulation: agentic (LangGraph) mode, 52 weeks per run, LLMMachine vs static BasicMachine
- Reproducibility: 5 runs per model with paired seeds `42..46` (every model faces the same seeded demand sequences)
- Providers: nan API (chat models via chat/completions) and OpenRouter (top-5 weekly `decisions` models via POST /api/alpha/decisions, adapted to a symmetric ±7.5%/±15% price-action menu per product)
- Exclusions: mimo-v2.6-flash (nan allowance exhausted), minimax-h3 (no key access), gemma4 (reasoning chains overflow the gateway timeout and its 5-request concurrency cap makes full runs unreliable)

## Final net profit per model (aggregate over runs)

| Model | N | Mean | ±σ | 95% CI | Median | Min | Max | Win rate |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `perplexity/pplx-decider-v1.1-27b` | 5 | 1,856.9 | 92.1 | [1,743, 1,971] | 1,880.4 | 1,728.9 | 1,966.3 | 100% (5/5) |
| `openai/gpt-6-luna-decisions` | 5 | 1,461.4 | 313.7 | [1,072, 1,851] | 1,450.7 | 1,024.7 | 1,872.3 | 100% (5/5) |
| `glm5.3-flash` | 5 | 1,289.0 | 409.6 | [781, 1,797] | 1,280.4 | 754.7 | 1,876.8 | 100% (5/5) |
| `deepseek-v4-flash` | 5 | 1,134.7 | 435.8 | [594, 1,676] | 1,167.3 | 553.5 | 1,749.3 | 100% (5/5) |
| `qwen3.8-flash` | 5 | 728.3 | 267.2 | [397, 1,060] | 737.8 | 312.0 | 1,034.1 | 80% (4/5) |
| `qwen3.6` | 5 | 443.0 | 35.9 | [399, 488] | 427.5 | 407.8 | 491.0 | 100% (5/5) |
| `typesafe/jev-1.13` | 5 | 277.4 | 33.9 | [235, 320] | 281.6 | 224.2 | 309.9 | 20% (1/5) |
| `inception/mercury-decide:free` | 5 | 277.1 | 35.7 | [233, 321] | 276.2 | 223.0 | 312.1 | 40% (2/5) |
| `cloudflare/clef-flash` | 5 | 269.4 | 92.9 | [154, 385] | 248.4 | 142.7 | 386.9 | 40% (2/5) |

## Head-to-head margin vs BasicMachine (LLM profit − Basic profit, final week)

| Model | Margin mean | ±σ | Min | Max | Wins |
|---|---:|---:|---:|---:|---:|
| `perplexity/pplx-decider-v1.1-27b` | 1,421.3 | 80.5 | 1,327.4 | 1,508.6 | 5/5 |
| `openai/gpt-6-luna-decisions` | 1,082.0 | 299.8 | 685.8 | 1,468.3 | 5/5 |
| `glm5.3-flash` | 946.0 | 375.8 | 457.4 | 1,490.3 | 5/5 |
| `deepseek-v4-flash` | 804.8 | 430.6 | 207.2 | 1,397.2 | 5/5 |
| `qwen3.8-flash` | 413.2 | 276.4 | -24.6 | 724.8 | 4/5 |
| `qwen3.6` | 139.7 | 54.4 | 68.2 | 200.0 | 5/5 |
| `typesafe/jev-1.13` | -19.7 | 36.0 | -65.4 | 22.4 | 1/5 |
| `inception/mercury-decide:free` | -19.4 | 37.3 | -66.7 | 20.2 | 2/5 |
| `cloudflare/clef-flash` | -23.8 | 87.9 | -118.4 | 100.9 | 2/5 |

## Per-run detail (final week)

| Model | Run | Seed | Basic profit | LLM profit | Margin | LLM avg price |
|---|---:|---:|---:|---:|---:|---:|
| `perplexity/pplx-decider-v1.1-27b` | 1 | 43 | 421.1 | 1,904.4 | 1,483.3 | 13.19 |
| `perplexity/pplx-decider-v1.1-27b` | 2 | 44 | 457.7 | 1,966.3 | 1,508.6 | 13.68 |
| `perplexity/pplx-decider-v1.1-27b` | 3 | 45 | 441.4 | 1,880.4 | 1,439.0 | 15.42 |
| `perplexity/pplx-decider-v1.1-27b` | 4 | 46 | 401.5 | 1,728.9 | 1,327.4 | 13.60 |
| `perplexity/pplx-decider-v1.1-27b` | 5 | 47 | 456.4 | 1,804.5 | 1,348.2 | 15.00 |
| `openai/gpt-6-luna-decisions` | 1 | 43 | 403.9 | 1,872.3 | 1,468.3 | 8.44 |
| `openai/gpt-6-luna-decisions` | 2 | 44 | 427.8 | 1,350.6 | 922.8 | 8.27 |
| `openai/gpt-6-luna-decisions` | 3 | 45 | 359.2 | 1,608.7 | 1,249.5 | 8.02 |
| `openai/gpt-6-luna-decisions` | 4 | 46 | 338.9 | 1,024.7 | 685.8 | 7.06 |
| `openai/gpt-6-luna-decisions` | 5 | 47 | 367.2 | 1,450.7 | 1,083.5 | 6.67 |
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
| `qwen3.6` | 1 | 43 | 308.1 | 427.5 | 119.3 | 1.71 |
| `qwen3.6` | 2 | 44 | 339.6 | 407.8 | 68.2 | 1.94 |
| `qwen3.6` | 3 | 45 | 295.9 | 418.5 | 122.6 | 1.62 |
| `qwen3.6` | 4 | 46 | 282.2 | 470.5 | 188.3 | 2.50 |
| `qwen3.6` | 5 | 47 | 291.0 | 491.0 | 200.0 | 1.95 |
| `typesafe/jev-1.13` | 1 | 43 | 303.0 | 302.1 | -0.9 | 1.62 |
| `typesafe/jev-1.13` | 2 | 44 | 334.7 | 269.3 | -65.4 | 1.62 |
| `typesafe/jev-1.13` | 3 | 45 | 288.4 | 281.6 | -6.7 | 1.62 |
| `typesafe/jev-1.13` | 4 | 46 | 272.1 | 224.2 | -47.9 | 1.62 |
| `typesafe/jev-1.13` | 5 | 47 | 287.6 | 309.9 | 22.3 | 1.62 |
| `inception/mercury-decide:free` | 1 | 43 | 302.3 | 312.1 | 9.8 | 1.60 |
| `inception/mercury-decide:free` | 2 | 44 | 334.4 | 267.7 | -66.7 | 1.62 |
| `inception/mercury-decide:free` | 3 | 45 | 288.1 | 276.2 | -11.9 | 1.65 |
| `inception/mercury-decide:free` | 4 | 46 | 271.6 | 223.0 | -48.6 | 1.63 |
| `inception/mercury-decide:free` | 5 | 47 | 286.2 | 306.4 | 20.2 | 1.64 |
| `cloudflare/clef-flash` | 1 | 43 | 302.1 | 327.8 | 25.8 | 1.84 |
| `cloudflare/clef-flash` | 2 | 44 | 332.2 | 248.4 | -83.8 | 1.82 |
| `cloudflare/clef-flash` | 3 | 45 | 286.1 | 386.9 | 100.9 | 1.69 |
| `cloudflare/clef-flash` | 4 | 46 | 261.1 | 142.7 | -118.4 | 1.29 |
| `cloudflare/clef-flash` | 5 | 47 | 284.6 | 241.2 | -43.4 | 1.59 |

## Notes on dispersion

- Coefficient of variation (σ/mean) of final profit: `perplexity/pplx-decider-v1.1-27b` 5.0%, `openai/gpt-6-luna-decisions` 21.5%, `glm5.3-flash` 31.8%, `deepseek-v4-flash` 38.4%, `qwen3.8-flash` 36.7%, `qwen3.6` 8.1%, `typesafe/jev-1.13` 12.2%, `inception/mercury-decide:free` 12.9%, `cloudflare/clef-flash` 34.5%
- Dispersion has two sources: (1) LLM sampling stochasticity (temperature 0.7, reasoning models) and (2) market randomness, which is seeded identically per run index across models (paired seeds), so differences between models are not explained by luckier demand draws.
- Best mean performer: `perplexity/pplx-decider-v1.1-27b` with 1,856.9 (95% CI ± 114.3), winning 5/5 runs against BasicMachine.
