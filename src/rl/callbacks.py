import os
import time
import json
import psutil
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import TrainerCallback

class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir, log_interval=10, sample_interval=50, plot_interval=10):
        self.output_dir = output_dir
        self.step_metrics = []
        self.start_time = None
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.plot_interval = plot_interval
        self.last_checkpoint_step = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "="*70)
        print("  🚀 TRAINING STARTED")
        print("="*70)
        print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Max steps: {args.max_steps}")
        print(f"🔄 Batch size: {args.per_device_train_batch_size}")
        print(f"📈 Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"🎲 Generations per prompt: {args.num_generations}")
        print(f"💾 Checkpoint every: {args.save_steps} steps")
        print(f"🖥️  Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
        
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"💾 Initial RAM usage: {mem_info.rss / 1024**3:.2f} GB")
        print("="*70 + "\n")
        
    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        elapsed = time.time() - self.start_time
        
        if state.log_history:
            latest_log = state.log_history[-1]
            
            metrics = {
                'step': current_step,
                'loss': latest_log.get('loss', None),
                'learning_rate': latest_log.get('learning_rate', None),
                'elapsed_time': elapsed,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            for key in latest_log:
                if 'reward' in key.lower():
                    metrics[key] = latest_log[key]
                    
            self.step_metrics.append(metrics)
            
            # Brief progress print
            progress_pct = (current_step / args.max_steps) * 100 if args.max_steps > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {current_step}/{args.max_steps} ({progress_pct:.1f}%) - ", end="")
            
            if metrics['loss'] is not None:
                print(f"Loss: {metrics['loss']:.4f}", end="")
            if 'rewards/mean' in metrics:
                print(f" | Reward: {metrics['rewards/mean']:.3f}", end="")
            
            if current_step > 0:
                avg_time_per_step = elapsed / current_step
                steps_remaining = args.max_steps - current_step
                eta_seconds = avg_time_per_step * steps_remaining
                eta_hours = eta_seconds / 3600
                print(f" | ETA: {eta_hours:.2f}h")
            else:
                print()
            
            # Plot loss every plot_interval steps
            if current_step % self.plot_interval == 0 and current_step > 0:
                self.plot_loss()
                
            # Detailed logging
            if current_step % self.log_interval == 0 or current_step == args.max_steps:
                self.log_detailed(metrics, elapsed)

    def plot_loss(self):
        steps = [m['step'] for m in self.step_metrics if m['loss'] is not None]
        losses = [m['loss'] for m in self.step_metrics if m['loss'] is not None]
        
        if not losses:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label='Training Loss')
        plt.title('Training Loss per Iteration')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(self.output_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        # print(f"📈 Loss plot updated: {plot_path}")

    def log_detailed(self, metrics, elapsed):
        print("\n" + "─"*70)
        print(f"📊 DETAILED METRICS - Step {metrics['step']}")
        print("─"*70)
        if metrics['loss'] is not None:
            print(f"📉 Loss: {metrics['loss']:.6f}")
        
        reward_keys = [k for k in metrics if 'reward' in k.lower()]
        if reward_keys:
            print("\n🎁 Reward Metrics:")
            for key in sorted(reward_keys):
                print(f"   {key}: {metrics[key]:.4f}")
        
        process = psutil.Process()
        print(f"\n💾 RAM Usage: {process.memory_info().rss / 1024**3:.2f} GB")
        print("─"*70 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n" + "="*70)
        print("  🎉 TRAINING COMPLETED")
        print("="*70)
        print(f"⏱️  Total duration: {total_time/3600:.2f}h")
        
        # Save final plot
        self.plot_loss()
        
        # Save metrics JSON
        metrics_file = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'total_time_seconds': total_time,
                'total_steps': state.global_step,
                'step_metrics': self.step_metrics
            }, f, indent=2)
        print(f"💾 Metrics saved to: {metrics_file}")
