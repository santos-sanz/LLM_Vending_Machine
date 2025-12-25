#!/usr/bin/env python3
"""
Test script for running the trained RL model from outputs/gemma_rl_vending_local
This script uses Ollama to run the model locally.
"""

import os
import sys

def main():
    print("=" * 60)
    print("Testing Trained RL Model: gemma_rl_vending_local")
    print("=" * 60)
    print()
    
    # Check if the model directory exists
    model_path = "outputs/gemma_rl_vending_local"
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        sys.exit(1)
    
    print(f"✓ Found model directory: {model_path}")
    print()
    
    # Instructions for setting up the model in Ollama
    print("SETUP INSTRUCTIONS:")
    print("-" * 60)
    print("To use this model with Ollama, you need to:")
    print()
    print("1. Create a Modelfile in the outputs/gemma_rl_vending_local directory:")
    print("   cat > outputs/gemma_rl_vending_local/Modelfile << 'EOF'")
    print("   FROM google/gemma-2b-it")
    print("   ADAPTER ./adapter_model.safetensors")
    print("   EOF")
    print()
    print("2. Create the model in Ollama:")
    print("   ollama create gemma-rl-vending -f outputs/gemma_rl_vending_local/Modelfile")
    print()
    print("3. Run this test again")
    print("-" * 60)
    print()
    
    # Check if user wants to proceed
    response = input("Have you completed the setup? (y/n): ").strip().lower()
    if response != 'y':
        print("\nPlease complete the setup steps above and run this script again.")
        sys.exit(0)
    
    print("\nRunning competitive simulation with trained RL model...")
    print("This may take several minutes depending on the number of weeks.")
    print()
    
    # Import and run the competition
    from src.competitive_run import run_competition
    
    # Run with local mode using the RL model
    result = run_competition(
        model_name="gemma-rl-vending",
        verbose=True,
        num_weeks=12,  # Start with 12 weeks for testing
        save_plot=True,
        record_history=True,
        mode="local"
    )
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  BasicMachine Profit: ${result['basic_profit']:.2f}")
    print(f"  LLMMachine Profit:   ${result['llm_profit']:.2f}")
    print(f"  Difference:          ${result['llm_profit'] - result['basic_profit']:.2f}")
    
    if result['llm_profit'] > result['basic_profit']:
        print(f"\n🎉 The RL-trained model WON by ${result['llm_profit'] - result['basic_profit']:.2f}!")
    else:
        print(f"\n📊 BasicMachine won by ${result['basic_profit'] - result['llm_profit']:.2f}")
    
    print()

if __name__ == "__main__":
    main()
