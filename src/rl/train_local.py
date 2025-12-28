import os
import sys
import argparse
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login

# Import our decoupled modules
from src.rl.config import (
    MODEL_NAME, OUTPUT_DIR, MAX_SEQ_LENGTH, 
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    GRADIENT_ACCUMULATION_STEPS, NUM_GENERATIONS, MAX_COMPLETION_LENGTH,
    LEARNING_RATE, WARMUP_STEPS, LOGGING_STEPS,
    HUGGINGFACE_TOKEN, HUGGINGFACE_ACCOUNT
)
from src.rl.data_utils import generate_prompt
from src.rl.rewards import format_reward_func, profit_reward_func, xml_reward_func
from src.rl.callbacks import MetricsCallback

# Set environment variable to suppress verbose simulation logs
os.environ["RL_TRAINING"] = "1"

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RL model for vending machine optimization')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of training steps (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Quick test run with minimal settings')
    parser.add_argument('--dataset-size', type=int, default=50,
                       help='Number of training examples to generate (default: 50)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size (default: 1)')
    parser.add_argument('--save-steps', type=int, default=None,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='Path to a checkpoint to resume training from')
    args = parser.parse_args()
    
    # Adjust settings for dry-run
    if args.dry_run:
        print("\n⚡ DRY RUN MODE - Using minimal settings\n")
        max_steps = 2
        dataset_size = 10
        save_steps = 1
        num_generations = 2
        num_sim_weeks = 5
    else:
        max_steps = args.steps
        dataset_size = args.dataset_size
        save_steps = args.save_steps if args.save_steps else max(1, max_steps // 2)
        num_generations = NUM_GENERATIONS
        num_sim_weeks = 5
    
    print("="*60)
    print("  RL TRAINING (REFACTORED)")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max steps: {max_steps}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Dataset
    print(f"Generating {dataset_size} training examples...")
    data = [generate_prompt(i) for i in range(dataset_size)]
    dataset = Dataset.from_list(data)
    print(f"✓ Dataset created\n")

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    # Authenticate
    if HUGGINGFACE_TOKEN:
        print("Authenticating with Hugging Face...")
        login(token=HUGGINGFACE_TOKEN)
        print("✓ Authenticated\n")

    # Model
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
    tokenizer.padding_side = "left" # Required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        device_map=device,
        token=HUGGINGFACE_TOKEN
    )
    
    # PEFT
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_rslora=True, # Stabilize higher rank adaptation
    )
    model = get_peft_model(model, peft_config)
    
    # Gradient tracking fixes
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()
    print()

    # Training Args
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=num_generations,
        generation_batch_size=num_generations,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        use_cpu=False,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        mask_truncated_completions=False,  # Critical: allows learning from truncated completions
    )

    # Callback
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_callback = MetricsCallback(output_dir=OUTPUT_DIR)

    # Trainer
    print("Initializing trainer...")
    
    # Wrap profit_reward_func to pass model and tokenizer for multi-turn simulation
    def interactive_profit_reward(*args, **kwargs):
        return profit_reward_func(*args, model=model, tokenizer=tokenizer, num_weeks=num_sim_weeks, **kwargs)
    interactive_profit_reward.__name__ = "profit_reward_func"
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[xml_reward_func, format_reward_func, interactive_profit_reward],
        args=training_args,
        train_dataset=dataset,
        callbacks=[metrics_callback]
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Hub upload
    if HUGGINGFACE_ACCOUNT and HUGGINGFACE_TOKEN:
        try:
            repo_name = f"{HUGGINGFACE_ACCOUNT}/vending-machine-rl-model"
            print(f"\nPushing to Hub: {repo_name}")
            model.push_to_hub(repo_name, token=HUGGINGFACE_TOKEN)
            tokenizer.push_to_hub(repo_name, token=HUGGINGFACE_TOKEN)
        except Exception as e:
            print(f"⚠️ Hub push failed: {e}")

if __name__ == "__main__":
    main()
