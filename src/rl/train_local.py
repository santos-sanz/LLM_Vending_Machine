import os
import random
import numpy as np
import json
import re
import argparse
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from dotenv import load_dotenv
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it"
OUTPUT_DIR = "outputs/gemma3_rl_vending_local"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16

# --- Product Class (Silent) ---
@dataclass
class Product:
    name: str
    price: float
    cost: float
    stock: int
    max_stock: int
    base_likelihood: float
    price_sensitivity: float

    @property
    def purchase_likelihood(self) -> float:
        calculated_likelihood = self.base_likelihood - (self.price * self.price_sensitivity)
        return max(0.0, min(1.0, calculated_likelihood))

# --- Vending Machine Class (Silent) ---
class VendingMachine:
    def __init__(self, initial_cash: float = 0.0, maintenance_cost: float = 20.0, initial_investment: float = 0.0):
        self.products: Dict[str, Product] = {}
        self.cash = initial_cash
        self.maintenance_cost = maintenance_cost
        self.initial_investment = initial_investment

    def add_product(self, config: dict):
        p = Product(**config)
        if p.name in self.products:
            existing = self.products[p.name]
            self.initial_investment += (p.stock - existing.stock) * p.cost
            self.products[p.name] = p
        else:
            self.products[p.name] = p
            self.initial_investment += p.stock * p.cost

    def get_product(self, name: str) -> Optional[Product]:
        return self.products.get(name)

    def recharge_products(self):
        for name, p in self.products.items():
            needed = p.max_stock - p.stock
            if needed > 0:
                cost = needed * p.cost
                self.cash -= cost
                p.stock = p.max_stock

    def sell_product(self, name: str) -> bool:
        p = self.products.get(name)
        if p and p.stock > 0:
            p.stock -= 1
            self.cash += p.price
            return True
        return False

    def apply_maintenance(self):
        self.cash -= self.maintenance_cost

    def calculate_profit_loss(self) -> float:
        return self.cash - self.initial_investment

# --- Simulation Engine (Silent) ---
PRODUCT_CONFIGS = [
    {"name": "Soda",      "price": 2.50, "cost": 1.00, "stock": 20, "max_stock": 20, "base_likelihood": 0.725, "price_sensitivity": 0.05},
    {"name": "Chips",     "price": 1.75, "cost": 0.75, "stock": 30, "max_stock": 30, "base_likelihood": 0.54,  "price_sensitivity": 0.08},
    {"name": "Candy Bar", "price": 1.25, "cost": 0.50, "stock": 40, "max_stock": 40, "base_likelihood": 0.75,  "price_sensitivity": 0.04},
    {"name": "Water",     "price": 1.00, "cost": 0.30, "stock": 50, "max_stock": 50, "base_likelihood": 0.83,  "price_sensitivity": 0.03}
]
CLIENT_LAMBDA = 25

def simulate_competitive_week(machines: List[VendingMachine]) -> Tuple[List[float], List[str]]:
    stockout_events = []
    reported_stockouts = set()

    for vm in machines:
        vm.recharge_products()
        vm.apply_maintenance()

    num_clients = np.random.poisson(CLIENT_LAMBDA)

    for client_idx in range(num_clients):
        # 1. Check for stockouts
        for i, vm in enumerate(machines):
            for p_name, p in vm.products.items():
                if p.stock == 0:
                    event_key = (i, p_name)
                    if event_key not in reported_stockouts:
                        msg = f"Machine {i}: {p_name} is out of stock!"
                        stockout_events.append(msg)
                        reported_stockouts.add(event_key)

        # 2. Gather options
        options = []
        for vm in machines:
            for p in vm.products.values():
                if p.stock > 0:
                    options.append((vm, p.name, p.purchase_likelihood))

        if not options:
            break

        # 3. Client Choice
        vms, names, likelihoods = zip(*options)
        total_likelihood = sum(likelihoods)

        if total_likelihood <= 0:
            continue

        probs = [l / total_likelihood for l in likelihoods]
        chosen_idx = np.random.choice(len(options), p=probs)
        chosen_vm, chosen_name, _ = options[chosen_idx]

        chosen_vm.sell_product(chosen_name)

    return [vm.calculate_profit_loss() for vm in machines], stockout_events

# --- Training Data Generation ---
def generate_random_market_data():
    return {
        "competitor_prices": {p["name"]: round(p["price"] * random.uniform(0.8, 1.2), 2) for p in PRODUCT_CONFIGS},
        "estimated_demand": random.choice(["Low", "Medium", "High"]),
        "marketing_intensity": random.choice(["Low", "Medium", "High"]),
        "upcoming_events": random.choice(["None", "local_festival", "school_holiday"])
    }

def generate_random_stockout_events():
    if random.random() < 0.7:
        return "No products ran out of stock this week."
    events = []
    num_events = random.randint(1, 3)
    for _ in range(num_events):
        prod = random.choice(PRODUCT_CONFIGS)["name"]
        events.append(f"Machine 0: {prod} is out of stock!")
    return "\n".join(events)

def generate_prompt(idx):
    market_data = generate_random_market_data()
    event_feedback = generate_random_stockout_events()
    week_num = random.randint(1, 10)

    prompt = f"""
    You are a Strategic Business Manager for 'LLMMachine'. You are in direct competition with 'BasicMachine'.
    Both machines share the same client pool. Clients choose the product with the highest 'purchase_likelihood'.

    IMPORTANT RULES:
    1. AT THE START OF EACH WEEK, YOUR MACHINE IS REFILLED TO MAX CAPACITY.
    2. 'BasicMachine' NEVER changes its prices.
    3. Your goal is to maximize NET PROFIT.

    STRATEGIC ADVICE:
    - Your priority is PROFIT, not just sales volume.
    - Selling out is NOT always the goal. If there is more supply than demand, you might never sell out.
    - If you raise prices, you might sell fewer items, but if the margin increase covers the volume loss, your PROFIT goes up.

    AVAILABLE TOOLS (Respond only in JSON):
    1. {{"action": "change_price", "parameters": {{"machine_name": "LLMMachine", "product_name": "...", "new_price": ...}}}}
    2. {{"action": "next_week"}} - to proceed.

    Provide your reasoning first, then the JSON block.

    --- WEEK {week_num} RESULTS ---
    Stockout Events:
    {event_feedback}

    Current Market Data:
    {json.dumps(market_data, indent=2)}

    Machines are being refilled NOW. What adjustments will you make for Week {week_num+1}?
    """
    return {"prompt": prompt}

# --- Reward Functions ---
def extract_json_action(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def format_reward_func(prompts, completions, **kwargs):
    """Reward function that checks if output contains valid JSON with an action key."""
    rewards = []
    for completion in completions:
        # In newer TRL, completions are strings, not dicts
        response_text = completion if isinstance(completion, str) else completion[0]["content"]
        action_json = extract_json_action(response_text)
        if action_json and "action" in action_json:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def profit_reward_func(prompts, completions, **kwargs):
    """Reward function that simulates the vending machine and returns profit."""
    rewards = []
    for completion in completions:
        # In newer TRL, completions are strings, not dicts
        response_text = completion if isinstance(completion, str) else completion[0]["content"]
        action_json = extract_json_action(response_text)

        if not action_json:
            rewards.append(-5.0)
            continue

        basic_machine = VendingMachine(initial_cash=0)
        llm_machine = VendingMachine(initial_cash=0)
        for config in PRODUCT_CONFIGS:
            basic_machine.add_product(config)
            llm_machine.add_product(config)

        valid_action = False
        if action_json.get("action") == "change_price":
            params = action_json.get("parameters", {})
            p_name = params.get("product_name")
            new_price = params.get("new_price")
            if p_name and new_price is not None:
                prod = llm_machine.get_product(p_name)
                if prod:
                    prod.price = float(new_price)
                    valid_action = True
        elif action_json.get("action") == "next_week":
             valid_action = True

        if not valid_action:
             rewards.append(-1.0)
             continue

        try:
            simulate_competitive_week([basic_machine, llm_machine])
            profit = llm_machine.calculate_profit_loss()
            rewards.append(profit)
        except Exception:
            rewards.append(-5.0)

    return rewards

# --- Custom Metrics Callback ---
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.step_metrics = []
        self.start_time = None
        self.format_rewards = []
        self.profit_rewards = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("  TRAINING STARTED")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Max steps: {args.max_steps}")
        print(f"Dataset size: {len(kwargs.get('train_dataloader', []))}")
        print("="*60 + "\n")
        
    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        elapsed = time.time() - self.start_time
        
        # Get current logs
        if state.log_history:
            latest_log = state.log_history[-1]
            
            # Extract metrics
            metrics = {
                'step': current_step,
                'loss': latest_log.get('loss', None),
                'learning_rate': latest_log.get('learning_rate', None),
                'elapsed_time': elapsed
            }
            
            # Store rewards if available
            if 'rewards/margins' in latest_log:
                metrics['reward_margin'] = latest_log['rewards/margins']
            if 'rewards/mean' in latest_log:
                metrics['reward_mean'] = latest_log['rewards/mean']
                
            self.step_metrics.append(metrics)
            
            # Print progress
            print(f"\n{'─'*60}")
            print(f"Step {current_step}/{args.max_steps} | Elapsed: {elapsed:.1f}s")
            if metrics['loss']:
                print(f"  Loss: {metrics['loss']:.4f}")
            if 'reward_mean' in metrics:
                print(f"  Avg Reward: {metrics['reward_mean']:.4f}")
            if 'reward_margin' in metrics:
                print(f"  Reward Margin: {metrics['reward_margin']:.4f}")
            if metrics['learning_rate']:
                print(f"  Learning Rate: {metrics['learning_rate']:.2e}")
                
            # Estimate time remaining
            steps_remaining = args.max_steps - current_step
            if current_step > 0:
                avg_time_per_step = elapsed / current_step
                eta = avg_time_per_step * steps_remaining
                print(f"  ETA: {eta/60:.1f} minutes")
            print(f"{'─'*60}")
            
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETED")
        print("="*60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {total_time/60:.2f} minutes ({total_time:.1f}s)")
        print(f"Total steps: {state.global_step}")
        print(f"Avg time per step: {total_time/max(state.global_step, 1):.2f}s")
        
        # Calculate statistics
        if self.step_metrics:
            losses = [m['loss'] for m in self.step_metrics if m['loss'] is not None]
            if losses:
                print(f"\nLoss Statistics:")
                print(f"  Initial: {losses[0]:.4f}")
                print(f"  Final: {losses[-1]:.4f}")
                print(f"  Mean: {np.mean(losses):.4f}")
                print(f"  Std: {np.std(losses):.4f}")
                print(f"  Min: {np.min(losses):.4f}")
                print(f"  Max: {np.max(losses):.4f}")
                
            rewards = [m.get('reward_mean') for m in self.step_metrics if m.get('reward_mean') is not None]
            if rewards:
                print(f"\nReward Statistics:")
                print(f"  Mean: {np.mean(rewards):.4f}")
                print(f"  Std: {np.std(rewards):.4f}")
                print(f"  Min: {np.min(rewards):.4f}")
                print(f"  Max: {np.max(rewards):.4f}")
                
        print("\n" + "="*60 + "\n")
        
        # Save metrics to file
        metrics_file = os.path.join(args.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'total_time_seconds': total_time,
                'total_steps': state.global_step,
                'step_metrics': self.step_metrics
            }, f, indent=2)
        print(f"Metrics saved to: {metrics_file}\n")

# --- Main Training Loop ---
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RL model for vending machine optimization')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of training steps (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Quick test run with minimal steps')
    parser.add_argument('--dataset-size', type=int, default=50,
                       help='Number of training examples to generate (default: 50)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--save-steps', type=int, default=None,
                       help='Save checkpoint every N steps (default: steps/2)')
    args = parser.parse_args()
    
    # Adjust settings for dry-run
    if args.dry_run:
        print("\n⚡ DRY RUN MODE - Using minimal settings for quick test\n")
        max_steps = 2
        dataset_size = 10
        save_steps = 1
    else:
        max_steps = args.steps
        dataset_size = args.dataset_size
        save_steps = args.save_steps if args.save_steps else max(1, max_steps // 2)
    
    print("="*60)
    print("  RL TRAINING CONFIGURATION")
    print("="*60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FULL TRAINING'}")
    print(f"Max steps: {max_steps}")
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every: {save_steps} steps")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Dataset
    print(f"Generating {dataset_size} training examples...")
    data = [generate_prompt(i) for i in range(dataset_size)]
    dataset = Dataset.from_list(data)
    print(f"✓ Dataset created with {len(dataset)} examples\n")

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    # Authenticate with Hugging Face if token is available
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        from huggingface_hub import login
        login(token=hf_token)
        print("✓ Authenticated\n")
    else:
        print("⚠️  No HUGGINGFACE_TOKEN found in .env")
        print("   Some models may require authentication\n")

    # Model
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        token=hf_token
    )
    print(f"✓ Model loaded\n")

    # PEFT Config
    print("Applying LoRA configuration...")
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()
    print()

    # Training Args
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-7,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_completion_length=128,
        max_steps=1200,
        save_steps=200,
        save_total_limit=1,
        report_to="none",
        use_cpu=False,
    )

    # Create callback
    metrics_callback = MetricsCallback()

    # Trainer
    print("Initializing trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, profit_reward_func],
        args=training_args,
        train_dataset=dataset,
        callbacks=[metrics_callback]
    )
    print("✓ Trainer ready\n")

    # Train
    trainer.train()

    # Save locally
    print(f"Saving to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Push to Hugging Face Hub
    hf_account = os.getenv("HUGGINGFACE_ACCOUNT")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if hf_account and hf_token:
        try:
            repo_name = f"{hf_account}/vending-machine-rl-model"
            print(f"\nPushing model to Hugging Face Hub: {repo_name}")
            
            model.push_to_hub(
                repo_name,
                token=hf_token,
                commit_message="Upload RL-trained vending machine model"
            )
            tokenizer.push_to_hub(
                repo_name,
                token=hf_token,
                commit_message="Upload tokenizer for RL-trained vending machine model"
            )
            
            print(f"✓ Model successfully pushed to: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"⚠️ Failed to push to Hugging Face: {e}")
            print("Model saved locally only.")
    else:
        print("⚠️ HUGGINGFACE_ACCOUNT or HUGGINGFACE_TOKEN not found in .env")
        print("Model saved locally only.")

if __name__ == "__main__":
    main()
