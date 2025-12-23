# LLM Vending Machine Simulation

A modular Python-based simulation of a vending machine operation with potential LLM integration via OpenRouter.

## Features

- **Modular Design**: Separated models, simulation logic, and configuration.
- **Weekly Simulation**: Simulates product recharging, maintenance costs, and varying client purchase attempts.
- **Profit/Loss Tracking**: Calculates and tracks financial performance over a 52-week period.
- **Data Visualization**: Generates a plot of weekly net profit using `matplotlib`.
- **LLM Ready**: Integrated client for OpenRouter to support future AI-driven features (e.g., dynamic pricing or personalized restocking).

## Project Structure

```
LLM_Vending_Machine/
├── src/
│   ├── models/           # Domain models (Product, VendingMachine)
│   ├── simulation/       # Simulation logic (Weekly simulation engine)
│   ├── llm/             # LLM client integration (OpenRouter)
│   ├── config.py         # Centralized configuration settings
│   └── main.py           # Simulation entry point
├── tests/                # Connectivity and logic tests
├── .env                  # Environment variables (API keys)
├── .gitignore            # Git exclusion rules
└── simulation_results.png # Generated simulation profit plot
```

## Getting Started

### Prerequisites

- Python 3.12+
- `pip`

### Installation

1. Install dependencies:
   ```bash
   pip install python-dotenv openai matplotlib
   ```

2. Create a `.env` file in the root directory and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

### Running the Simulation

Execute the main simulation script:
```bash
python3 src/main.py
```
This will run a 52-week simulation and save the resulting profit trend in `simulation_results.png`.

### Verifying LLM Connectivity

Run the test script to verify your OpenRouter configuration:
```bash
PYTHONPATH=. python3 tests/test_llm.py
```

## License

MIT