# Local and Online Model Support - Usage Guide

This document explains how to use the competitive vending machine simulation with both local (Ollama) and online (OpenRouter) models.

## Quick Start

### Online Mode (OpenRouter)

Run with any OpenRouter model (default mode):

```bash
python3 src/competitive_run.py --model nvidia/nemotron-3-nano-30b-a3b:free --weeks 12
```

### Local Mode (Ollama)

Run with a local Ollama model:

```bash
python3 src/competitive_run.py --mode local --local-model qwen --weeks 12
```

## Using Your Trained RL Model

### Step 1: Set up the model in Ollama

Create a Modelfile for your trained model:

```bash
cat > outputs/gemma_rl_vending_local/Modelfile << 'EOF'
FROM google/gemma-2b-it
ADAPTER ./adapter_model.safetensors
EOF
```

### Step 2: Build the model in Ollama

```bash
ollama create gemma-rl-vending -f outputs/gemma_rl_vending_local/Modelfile
```

### Step 3: Run the test script

```bash
python3 test_rl_model.py
```

Or manually run the simulation:

```bash
python3 src/competitive_run.py --mode local --local-model gemma-rl-vending --weeks 52
```

## Command-Line Options

- `--mode`: Choose between `local` (Ollama) or `online` (OpenRouter). Default: `online`
- `--model`: Model name for online mode (e.g., `nvidia/nemotron-3-nano-30b-a3b:free`)
- `--local-model`: Model name for local mode (e.g., `qwen`, `gemma-rl-vending`)
- `--weeks`: Number of weeks to simulate (default: 52)
- `--verbose`: Enable detailed output
- `--json-output`: Output results as JSON

## Examples

### Compare Online vs Local Models

```bash
# Online with Nvidia model
python3 src/competitive_run.py --mode online --model nvidia/nemotron-3-nano-30b-a3b:free --weeks 12

# Local with Qwen
python3 src/competitive_run.py --mode local --local-model qwen --weeks 12

# Local with your trained RL model
python3 src/competitive_run.py --mode local --local-model gemma-rl-vending --weeks 12
```

### Run Benchmarks

Both modes are compatible with the existing benchmark system in `src/benchmark.py`.

## Configuration

You can set default models via environment variables in your `.env` file:

```env
# Online model settings
DEFAULT_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
OPENROUTER_API_KEY=your_key_here

# Local model settings
DEFAULT_LOCAL_MODEL=qwen
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Troubleshooting

### Ollama Connection Issues

If you get connection errors with Ollama:

1. Ensure Ollama is running: `ollama serve`
2. Check that models are installed: `ollama list`
3. Verify the base URL in your `.env` file or use default `http://localhost:11434/v1`

### Model Not Found

If Ollama can't find your model:

```bash
# List available models
ollama list

# Pull a model if needed
ollama pull qwen

# Or create your custom model as described above
ollama create gemma-rl-vending -f outputs/gemma_rl_vending_local/Modelfile
```
