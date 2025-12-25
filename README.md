# LLM Vending Machine Simulation

A modular Python-based simulation of a vending machine operation where multiple AI models compete to optimize pricing and maximize profit.

## 🚀 Overview

This project simulates the economic environment of vending machines. It provides a platform to test how different Strategic Business Managers (implemented via LLMs) can adapt to market competition, stockouts, and price sensitivity.

## ✨ Features

- **Multi-Model Competition**: Run simulations where different LLMs (e.g., Mistral, DeepSeek, Google) control competing machines.
- **Dynamic Pricing**: LLMs use tool-calling to adjust prices weekly based on market data and competitor performance.
- **Realistic Econ-Sim**: Includes maintenance costs, price sensitivity, base likelihood of purchase, and stock management.
- **Persistence & Analytics**: Results are stored in CSV formats, and performance trends are visualized via automatically generated graphs.
- **Modular Architecture**: Clean separation between domain models, simulation logic, and AI integration.

## 🏗️ Project Structure

```text
LLM_Vending_Machine/
├── src/
│   ├── models/           # Domain models: Product, VendingMachine
│   ├── simulation/       # Simulation engines for basic and competitive scenarios
│   ├── llm/              # OpenRouter client and decision-making tools
│   ├── utils/            # Shared helpers for plotting and data processing
│   ├── config.py         # Centralized configuration (Models, Products, PnL settings)
│   ├── main.py           # Single-machine baseline simulation
│   ├── competitive_run.py # Baseline vs. LLM competition script
│   ├── multi_model_run.py # Multi-LLM competition script
│   └── benchmark.py       # LLM performance benchmarking pipeline
├── data/results/         # Simulation history and persistence records
├── tests/                # Unit and connectivity tests
├── .env                  # Environment variables (OpenRouter API Key)
└── *.png                 # Generated performance visualizations
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.12+
- [OpenRouter API Key](https://openrouter.ai/)

### Installation

1. **Clone and Install Dependencies**:
   ```bash
   pip install python-dotenv openai matplotlib
   ```

2. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_key_here
   DEFAULT_MODEL=mistralai/mistral-7b-instruct:free
   ```

### Running Simulations

The project includes four main entry points depending on the complexity you want to simulate:

1. **Basic Baseline**:
   ```bash
   python3 src/main.py
   ```
   *Runs a single machine with fixed prices to establish a baseline.*

2. **Human vs. AI (Baseline vs. LLM)**:
   ```bash
   python3 src/competitive_run.py
   ```
   *One LLM-controlled machine competes against a fixed-price 'BasicMachine'.*

3. **AI vs. AI (Multi-Model)**:
   ```bash
   python3 src/multi_model_run.py
   ```
   *Two different LLM models compete in the same market.*

4. **Benchmarking Pipeline**:
   ```bash
   python3 src/benchmark.py
   ```
   *Runs multiple simulations across various models and generates a performance summary table.*

## 📊 Monitoring Results

- **Graphs**: Every run generates a `.png` file comparing weekly profits.
- **Logs**: Detailed market data and LLM reasoning are printed to the console during simulation.
- **History**: Check `data/results/simulation_history.csv` for a longitudinal record of all simulation runs.

## 🛡️ License

MIT