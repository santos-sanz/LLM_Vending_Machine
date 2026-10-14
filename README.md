# LLM Vending Machine Simulation

A modular Python-based simulation where **Large Language Models (LLMs)** act as autonomous economic agents, competing to optimize pricing and maximize profit in a dynamic market environment.

> **New in v2.0**: Now featuring an **Agentic Mode** powered by [LangGraph](https://langchain-ai.github.io/langgraph/), enabling distinct reasoning, tool usage, and state management steps for the AI agents.

## 🚀 Overview

This project simulates a competitive vending machine market. It provides a platform to test how different "Strategic Business Manager" personas (implemented via LLMs) adapt to:
- **Competitor Actions**: Surviving against a fixed-strategy "BasicMachine" or other LLMs.
- **Market Dynamics**: Reacting to price sensitivity, random demand fluctuations, and stockouts.
- **Long-term Planning**: Balancing immediate sales vs. profit margins over a 52-week simulation.

## ✨ Features

- **🧠 Agentic Architecture**: Uses LangGraph to model the agent's workflow: `Simulate Market` -> `Reasoning (LLM)` -> `Tool Execution` -> `Next Week`.
- **⚔️ Multi-Model Competition**: Pit top models (Mistral, DeepSeek, GPT-4o, etc.) against each other or against baseline algorithms.
- **🛠️ Tool Use**: Agents have discrete tools to `change_price(product, new_price)` and inspect `get_market_data()`.
- **📈 Realistic Econ-Sim**: Includes product elasticity, maintenance costs, restocking mechanics, and cumulative profit tracking.
- **📊 Analytics & Benchmarking**: 
  - Automated profit plotting (matplotlib).
  - Detailed CSV logs of every decision and market event.
  - Benchmarking pipeline to run N trials and calculate win rates.

## 🏗️ Project Structure

```text
LLM_Vending_Machine/
├── src/
│   ├── agentic_run.py    # 🧠 Agent Run: LangGraph-based agent simulation (Recommended)
│   ├── competitive_run.py # 📜 Legacy: Standard loop simulation
│   ├── benchmark.py       # 📊 Pipeline to run multiple trials & aggregate stats
│   ├── multi_model_run.py # ⚔️ AI vs AI competition script
│   ├── main.py            # 🟢 Simple single-machine baseline
│   ├── config.py          # ⚙️ Configuration (Products, Costs, API Keys)
│   ├── models/            # 📦 Domain Objects: Product, VendingMachine
│   ├── simulation/        # 🎲 Market Engine & Physics
│   ├── llm/               # 🤖 LLM Client & Tool Definitions
│   └── utils/             # 📉 Plotting & Helper functions
├── data/
│   ├── benchmarks/        # 📂 Raw & Summary CSVs from benchmarks
│   ├── results/           # 📂 Global simulation history
│   └── logs/              # 📝 Detailed weekly logs
├── .env                   # 🔑 Secrets (API Keys)
└── *.png                  # 🖼️ Generated profit graphs
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.12+
- [OpenRouter API Key](https://openrouter.ai/) (or OpenAI compatible key)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/santos-sanz/LLM_Vending_Machine.git
   cd LLM_Vending_Machine
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install python-dotenv openai matplotlib langgraph langchain-openai langchain-core
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   # nan API (default provider, OpenAI-compatible)
   NAN_API_KEY=sk-your-nan-key
   NAN_BASE_URL=https://api.nan.builders/v1
   # OpenRouter (for "decisions" modality models)
   OPENROUTER_API_KEY=sk-or-your-key-here
   DEFAULT_MODEL=deepseek-v4-flash
   ```

## 🎮 Running Simulations

The project supports multiple modes depending on what you want to test.

### 1. Agentic Competitive Run (Recommended)
Runs the **LangGraph** agent against the Baseline Machine. This mode allows the agent to "think", use tools, and loop until it decides to proceed to the next week.

```bash
python3 src/agentic_run.py --model mistralai/mistral-large-latest --weeks 52
```

### 2. Standard Benchmarking
Run a batch of simulations to statistically validate a model's performance. Each model runs N times with **paired seeds** (every model faces the same seeded demand sequences), and results are aggregated with dispersion statistics (mean, σ, min/max, 95% CI, win rate).

```bash
# Benchmark all configured models (nan chat models + OpenRouter decision models):
python3 src/benchmark.py --runs 5 --weeks 52 --workers 20

# Or a custom subset:
python3 src/benchmark.py --models "deepseek-v4-flash,typesafe/jev-1.13" --runs 5 --seed 42

# Aggregate a raw CSV into a dispersion report (Markdown + console tables):
python3 scripts/analyze_benchmark.py --raw data/benchmarks/benchmark_raw_<ts>.csv

# Merge shards from multiple runs before analysis:
python3 scripts/merge_benchmarks.py --inputs raw1.csv,raw2.csv --output combined.csv
```

**Model providers**:
- **nan API (chat models)**: `deepseek-v4-flash`, `qwen3.8-flash`, `glm5.3-flash`, `qwen3.6` — free-form JSON tool calling over `chat/completions`. "Thinking" models are bounded via per-model generation overrides to avoid gateway timeouts.
- **OpenRouter (decision models)**: the top-10 weekly `decisions` modality models (e.g. `typesafe/jev-1.13`, `perplexity/pplx-decider-v1.1-27b`, `openai/gpt-6-luna-decisions`, `cloudflare/clef-flash`, `inception/mercury-decide:free`, `nace-ai/drex-v1.5`, `microsoft/microsoft-decision-1`, `liquid/d1`, `cloudflare/clef`, `upstage/solar-decide`). They do not support `chat/completions`; the agent drives them through `POST /api/alpha/decisions` with one focused `choice` question per product (symmetric price-action menu: keep / ±7.5% / ±15%) and a **flat product-scoped state** (price, cost, competitor price, stock, stockouts) — nested full-market JSON makes these models default to "keep". Benchmark reports tag each model with its `model_type` (`decider` vs `standard`).

### 3. Legacy One-vs-One
The original simulation loop without LangGraph. Good for quick debugging.

```bash
python3 src/competitive_run.py --model nex-agi/deepseek-v3 --weeks 20
```

### 4. AI vs AI (Battle Mode)
Two different LLMs compete in the same market.

```bash
python3 src/multi_model_run.py
```

## 🧩 How It Works

1. **Initialization**: Two machines (`BasicMachine` & `LLMMachine`) start with 0 cash and full stock.
2. **The Week Loop**:
   - **Refill**: Machines are restocked to capacity.
   - **Agent Turn**: 
     - The Agent receives last week's sales data, stockouts, and current pricing.
     - It reasons about the market (e.g., "I sold out too fast, I should raise prices").
     - It executes tools to update prices.
   - **Simulation**: The engine runs 7 virtual days. Customers choose products based on `Purchase Probability ~ (Base Utility / Price)`.
   - **Accounting**: Profits are calculated (Revenue - Cost of Goods - Maintenance).
3. **Winner**: After N weeks, the machine with the highest customizable **Net Profit** wins.

## 📊 Monitoring

- **Terminal**: Real-time "Thinking..." logs from the agent and weekly profit summaries.
- **Images**: Check the root (or `images/` folder) for `profit_comparison_*.png` plots.
- **Data**: `data/results/simulation_results.csv` contains the high-level metrics of every run.

## 🛡️ License

MIT License. Free to use and modify.