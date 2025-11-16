"""
This module implements various pruning techniques to complement torch.compile
optimization by reducing model size and inference cost.

Pruning Techniques Implemented:
- Random Weight Pruning
- Magnitude-based Weight Pruning
- Random Neuron Pruning (Structured)
- Magnitude-based Neuron Pruning (Structured)
- Fisher Information-based Pruning
- Hessian-based Pruning

Pruning Schedules:
- One-shot Before Training
- One-shot After Training (with fine-tuning)
- Iterative (Gradual) Pruning
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
import ray
import json
import os
from datetime import datetime
from enum import Enum
import wandb

# ============================================================
# Configuration
# ============================================================
class PruningTechnique(Enum):
    """Pruning techniques for selecting which weights/neurons to remove"""
    NONE = "none"
    RANDOM_WEIGHT = "random_weight"
    MAGNITUDE_WEIGHT = "magnitude_weight"
    RANDOM_NEURON = "random_neuron"
    MAGNITUDE_NEURON = "magnitude_neuron"
    FISHER_WEIGHT = "fisher_weight"           
    FISHER_NEURON = "fisher_neuron"           
    HESSIAN_WEIGHT = "hessian_weight"         
    HESSIAN_NEURON = "hessian_neuron"        

class PruningSchedule(Enum):
    """When to apply pruning during training"""
    ONESHOT_BEFORE = "oneshot_before"   # Prune once before training
    ONESHOT_AFTER = "oneshot_after"     # Train, prune, then fine-tune
    ITERATIVE = "iterative"             # Gradually prune during training

# ============================================================
# Custom Pruned Policy Network
# ============================================================
class PrunedPolicyNet(TorchModelV2, nn.Module):
    """
    Custom policy network with pruning support.
    Supports both unstructured (weight-level) and structured (neuron-level) pruning.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        in_dim = int(np.product(obs_space.shape))
        hidden_dim = 64
        
        # Network layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_outputs)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Pruning masks
        self.register_buffer('mask_fc1_weight', torch.ones_like(self.fc1.weight))
        self.register_buffer('mask_fc2_weight', torch.ones_like(self.fc2.weight))
        self.register_buffer('mask_fc1_neuron', torch.ones(hidden_dim))
        self.register_buffer('mask_fc2_neuron', torch.ones(hidden_dim))
        
        # Fisher information and Hessian storage
        self.register_buffer('fisher_fc1', torch.zeros_like(self.fc1.weight))
        self.register_buffer('fisher_fc2', torch.zeros_like(self.fc2.weight))
        self.register_buffer('hessian_fc1', torch.zeros_like(self.fc1.weight))
        self.register_buffer('hessian_fc2', torch.zeros_like(self.fc2.weight))
        
        self._value_out = None
        self.pruning_mode = "neuron"  # "weight" or "neuron"
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.float()
        
        if self.pruning_mode == "weight":
            # Weight-level pruning
            masked_weight1 = self.fc1.weight * self.mask_fc1_weight
            x = F.linear(obs, masked_weight1, self.fc1.bias)
            x = F.relu(x)
            
            masked_weight2 = self.fc2.weight * self.mask_fc2_weight
            x = F.linear(x, masked_weight2, self.fc2.bias)
            x = F.relu(x)
        else:
            # Neuron-level pruning (structured)
            x = F.relu(self.fc1(obs))
            x = x * self.mask_fc1_neuron
            
            x = F.relu(self.fc2(x))
            x = x * self.mask_fc2_neuron
        
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        self._value_out = value.view(-1)
        return logits, state
    
    def value_function(self):
        return self._value_out
    
    # ==================== Fisher Information ====================
    
    def compute_fisher_information(self, samples, num_samples=1000):
        """
        Compute Fisher Information Matrix approximation.
        Fisher Information measures the importance of each weight.
        """
        self.fisher_fc1.zero_()
        self.fisher_fc2.zero_()
        
        self.eval()
        for i in range(min(num_samples, len(samples))):
            # Get a sample
            obs = torch.from_numpy(samples[i]).float().unsqueeze(0)
            
            # Forward pass
            if self.pruning_mode == "weight":
                masked_weight1 = self.fc1.weight * self.mask_fc1_weight
                x = F.linear(obs, masked_weight1, self.fc1.bias)
                x = F.relu(x)
                masked_weight2 = self.fc2.weight * self.mask_fc2_weight
                x = F.linear(x, masked_weight2, self.fc2.bias)
                x = F.relu(x)
            else:
                x = F.relu(self.fc1(obs))
                x = x * self.mask_fc1_neuron
                x = F.relu(self.fc2(x))
                x = x * self.mask_fc2_neuron
            
            logits = self.policy_head(x)
            
            # Sample action from policy
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute gradient of log probability
            for action_idx in range(logits.shape[-1]):
                if probs[0, action_idx] > 0.01:  # Only for likely actions
                    self.zero_grad()
                    log_prob = log_probs[0, action_idx] * probs[0, action_idx]
                    log_prob.backward(retain_graph=True)
                    
                    # Accumulate squared gradients (Fisher Information)
                    if self.fc1.weight.grad is not None:
                        self.fisher_fc1 += self.fc1.weight.grad.data ** 2
                    if self.fc2.weight.grad is not None:
                        self.fisher_fc2 += self.fc2.weight.grad.data ** 2
        
        # Average
        self.fisher_fc1 /= num_samples
        self.fisher_fc2 /= num_samples
        self.train()
    
    # ==================== Hessian Approximation ====================
    
    def compute_hessian_approximation(self, samples, num_samples=1000):
        """
        Compute diagonal Hessian approximation.
        Hessian captures second-order sensitivity of the loss.
        """
        self.hessian_fc1.zero_()
        self.hessian_fc2.zero_()
        
        self.eval()
        for i in range(min(num_samples, len(samples))):
            obs = torch.from_numpy(samples[i]).float().unsqueeze(0)
            
            # Forward pass
            if self.pruning_mode == "weight":
                masked_weight1 = self.fc1.weight * self.mask_fc1_weight
                x = F.linear(obs, masked_weight1, self.fc1.bias)
                x = F.relu(x)
                masked_weight2 = self.fc2.weight * self.mask_fc2_weight
                x = F.linear(x, masked_weight2, self.fc2.bias)
                x = F.relu(x)
            else:
                x = F.relu(self.fc1(obs))
                x = x * self.mask_fc1_neuron
                x = F.relu(self.fc2(x))
                x = x * self.mask_fc2_neuron
            
            logits = self.policy_head(x)
            value = self.value_head(x)
            
            # Compute loss (simplified: just value prediction)
            loss = value.sum()
            
            # First gradient
            self.zero_grad()
            loss.backward(create_graph=True)
            
            # Second gradient (Hessian diagonal)
            if self.fc1.weight.grad is not None:
                grad_fc1 = self.fc1.weight.grad.clone()
                self.zero_grad()
                grad_fc1.sum().backward()
                if self.fc1.weight.grad is not None:
                    self.hessian_fc1 += torch.abs(self.fc1.weight.grad.data)
            
            if self.fc2.weight.grad is not None:
                grad_fc2 = self.fc2.weight.grad.clone()
                self.zero_grad()
                grad_fc2.sum().backward()
                if self.fc2.weight.grad is not None:
                    self.hessian_fc2 += torch.abs(self.fc2.weight.grad.data)
        
        # Average
        self.hessian_fc1 /= num_samples
        self.hessian_fc2 /= num_samples
        self.train()
    
    # ==================== Pruning Methods ====================
    
    def apply_random_weight_pruning(self, prune_ratio):
        """Randomly prune individual weights"""
        self.pruning_mode = "weight"
        self.mask_fc1_weight = torch.bernoulli(torch.ones_like(self.fc1.weight) * (1 - prune_ratio))
        self.mask_fc2_weight = torch.bernoulli(torch.ones_like(self.fc2.weight) * (1 - prune_ratio))
    
    def apply_magnitude_weight_pruning(self, prune_ratio):
        """Prune weights with smallest magnitude"""
        self.pruning_mode = "weight"
        
        weights1 = self.fc1.weight.data.abs()
        weights2 = self.fc2.weight.data.abs()
        
        all_weights = torch.cat([weights1.flatten(), weights2.flatten()])
        threshold = torch.quantile(all_weights, prune_ratio)
        
        self.mask_fc1_weight = (weights1 > threshold).float()
        self.mask_fc2_weight = (weights2 > threshold).float()
    
    def apply_fisher_weight_pruning(self, prune_ratio):
        """Prune weights with lowest Fisher Information (least important)"""
        self.pruning_mode = "weight"
        
        # Combine Fisher information from both layers
        all_fisher = torch.cat([self.fisher_fc1.flatten(), self.fisher_fc2.flatten()])
        threshold = torch.quantile(all_fisher, prune_ratio)
        
        # Keep weights with high Fisher information
        self.mask_fc1_weight = (self.fisher_fc1 > threshold).float()
        self.mask_fc2_weight = (self.fisher_fc2 > threshold).float()
    
    def apply_hessian_weight_pruning(self, prune_ratio):
        """Prune weights with lowest Hessian (least sensitive to loss)"""
        self.pruning_mode = "weight"
        
        all_hessian = torch.cat([self.hessian_fc1.flatten(), self.hessian_fc2.flatten()])
        threshold = torch.quantile(all_hessian, prune_ratio)
        
        self.mask_fc1_weight = (self.hessian_fc1 > threshold).float()
        self.mask_fc2_weight = (self.hessian_fc2 > threshold).float()
    
    def apply_random_neuron_pruning(self, prune_ratio):
        """Randomly prune entire neurons (structured)"""
        self.pruning_mode = "neuron"
        self.mask_fc1_neuron = torch.bernoulli(torch.ones(64) * (1 - prune_ratio))
        self.mask_fc2_neuron = torch.bernoulli(torch.ones(64) * (1 - prune_ratio))
    
    def apply_magnitude_neuron_pruning(self, prune_ratio):
        """Prune neurons with smallest L2 norm (structured)"""
        self.pruning_mode = "neuron"
        
        norm1 = torch.norm(self.fc1.weight.data, p=2, dim=1)
        norm2 = torch.norm(self.fc2.weight.data, p=2, dim=1)
        
        threshold1 = torch.quantile(norm1, prune_ratio)
        threshold2 = torch.quantile(norm2, prune_ratio)
        
        self.mask_fc1_neuron = (norm1 > threshold1).float()
        self.mask_fc2_neuron = (norm2 > threshold2).float()
    
    def apply_fisher_neuron_pruning(self, prune_ratio):
        """Prune neurons with lowest total Fisher Information"""
        self.pruning_mode = "neuron"
        
        # Sum Fisher information over input dimensions for each neuron
        fisher1_per_neuron = self.fisher_fc1.sum(dim=1)  # [64]
        fisher2_per_neuron = self.fisher_fc2.sum(dim=1)  # [64]
        
        threshold1 = torch.quantile(fisher1_per_neuron, prune_ratio)
        threshold2 = torch.quantile(fisher2_per_neuron, prune_ratio)
        
        self.mask_fc1_neuron = (fisher1_per_neuron > threshold1).float()
        self.mask_fc2_neuron = (fisher2_per_neuron > threshold2).float()
    
    def apply_hessian_neuron_pruning(self, prune_ratio):
        """Prune neurons with lowest total Hessian"""
        self.pruning_mode = "neuron"
        
        hessian1_per_neuron = self.hessian_fc1.sum(dim=1)
        hessian2_per_neuron = self.hessian_fc2.sum(dim=1)
        
        threshold1 = torch.quantile(hessian1_per_neuron, prune_ratio)
        threshold2 = torch.quantile(hessian2_per_neuron, prune_ratio)
        
        self.mask_fc1_neuron = (hessian1_per_neuron > threshold1).float()
        self.mask_fc2_neuron = (hessian2_per_neuron > threshold2).float()
    
    def get_sparsity(self):
        """Return current sparsity level"""
        if self.pruning_mode == "weight":
            total = self.mask_fc1_weight.numel() + self.mask_fc2_weight.numel()
            active = self.mask_fc1_weight.sum().item() + self.mask_fc2_weight.sum().item()
        else:
            total = 128
            active = self.mask_fc1_neuron.sum().item() + self.mask_fc2_neuron.sum().item()
        return 1.0 - (active / total)
    
    def get_layer_sparsity(self):
        """Return sparsity per layer"""
        if self.pruning_mode == "weight":
            sparsity_fc1 = 1.0 - (self.mask_fc1_weight.sum() / self.mask_fc1_weight.numel()).item()
            sparsity_fc2 = 1.0 - (self.mask_fc2_weight.sum() / self.mask_fc2_weight.numel()).item()
        else:
            sparsity_fc1 = 1.0 - (self.mask_fc1_neuron.sum() / 64).item()
            sparsity_fc2 = 1.0 - (self.mask_fc2_neuron.sum() / 64).item()
        return sparsity_fc1, sparsity_fc2

ModelCatalog.register_custom_model("pruned_policy", PrunedPolicyNet)

# ============================================================
# Pruning Experiment Runner
# ============================================================
class PruningExperiment:
    """
    Manages a single pruning experiment with comprehensive logging.
    """
    
    def __init__(self, 
                 technique=PruningTechnique.NONE,
                 schedule=PruningSchedule.ONESHOT_BEFORE,
                 prune_ratio=0.0,
                 exp_name="baseline",
                 use_wandb=True,
                 wandb_project="rl-pruning",
                 log_dir="pruning_logs"):
        
        self.technique = technique
        self.schedule = schedule
        self.prune_ratio = prune_ratio
        self.exp_name = exp_name
        self.use_wandb = use_wandb
        
        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=exp_name,
                config={
                    "technique": technique.value,
                    "schedule": schedule.value,
                    "target_prune_ratio": prune_ratio,
                    "environment": "CartPole-v1",
                    "algorithm": "PPO",
                    "hidden_dim": 64,
                    "train_batch_size": 4000,
                    "learning_rate": 3e-4,
                },
                reinit=True
            )
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{exp_name}_{timestamp}.jsonl")
        
        # Build RLlib PPO algorithm
        config = (
            PPOConfig()
            .environment(env="CartPole-v1")
            .framework("torch")
            .rollouts(num_rollout_workers=2)
            .training(
                model={"custom_model": "pruned_policy"},
                train_batch_size=4000,
                lr=3e-4,
            )
            .resources(num_gpus=0)
        )
        
        self.algo = config.build()
        self.policy = self.algo.get_policy()
        self.stats = []
        self.logf = open(self.log_path, "w")
        
        # Apply pruning based on schedule
        if schedule == PruningSchedule.ONESHOT_BEFORE and technique != PruningTechnique.NONE:
            self._apply_pruning(self.prune_ratio)
    
    def _collect_samples_for_importance(self, num_samples=1000):
        """Collect samples from environment for Fisher/Hessian computation"""
        samples = []
        env = gym.make("CartPole-v1")
        
        for _ in range(num_samples):
            obs, _ = env.reset()
            samples.append(obs)
            
            # Also collect some during episode
            for _ in range(10):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                samples.append(obs)
                if terminated or truncated:
                    break
        
        env.close()
        return np.array(samples[:num_samples])
    
    def _apply_pruning(self, current_ratio=None):
        """Apply pruning based on technique"""
        if current_ratio is None:
            current_ratio = self.prune_ratio
        
        model = self.policy.model
        
        # Compute importance scores if needed
        if self.technique in [PruningTechnique.FISHER_WEIGHT, PruningTechnique.FISHER_NEURON]:
            print(f"  Computing Fisher Information...")
            samples = self._collect_samples_for_importance()
            model.compute_fisher_information(samples)
        
        if self.technique in [PruningTechnique.HESSIAN_WEIGHT, PruningTechnique.HESSIAN_NEURON]:
            print(f"  Computing Hessian approximation...")
            samples = self._collect_samples_for_importance()
            model.compute_hessian_approximation(samples)
        
        # Apply pruning
        if self.technique == PruningTechnique.RANDOM_WEIGHT:
            model.apply_random_weight_pruning(current_ratio)
        elif self.technique == PruningTechnique.MAGNITUDE_WEIGHT:
            model.apply_magnitude_weight_pruning(current_ratio)
        elif self.technique == PruningTechnique.FISHER_WEIGHT:
            model.apply_fisher_weight_pruning(current_ratio)
        elif self.technique == PruningTechnique.HESSIAN_WEIGHT:
            model.apply_hessian_weight_pruning(current_ratio)
        elif self.technique == PruningTechnique.RANDOM_NEURON:
            model.apply_random_neuron_pruning(current_ratio)
        elif self.technique == PruningTechnique.MAGNITUDE_NEURON:
            model.apply_magnitude_neuron_pruning(current_ratio)
        elif self.technique == PruningTechnique.FISHER_NEURON:
            model.apply_fisher_neuron_pruning(current_ratio)
        elif self.technique == PruningTechnique.HESSIAN_NEURON:
            model.apply_hessian_neuron_pruning(current_ratio)
        
        # Log pruning event
        if self.use_wandb:
            sparsity_fc1, sparsity_fc2 = model.get_layer_sparsity()
            wandb.log({
                "pruning_event/ratio": current_ratio,
                "pruning_event/actual_sparsity": model.get_sparsity(),
                "pruning_event/fc1_sparsity": sparsity_fc1,
                "pruning_event/fc2_sparsity": sparsity_fc2,
            })
    
    def _log(self, record):
        """Log record to file"""
        json.dump(record, self.logf)
        self.logf.write("\n")
        self.logf.flush()
    
    def train(self, num_epochs=30, iterative_prune_epochs=None):
        """
        Train the model with specified pruning schedule.
        
        Args:
            num_epochs: Total training epochs
            iterative_prune_epochs: List of epochs to apply pruning (for iterative schedule)
        """
        print(f"\n{'='*70}")
        print(f"Experiment: {self.exp_name}")
        print(f"Technique: {self.technique.value} | Schedule: {self.schedule.value}")
        print(f"Target Prune Ratio: {self.prune_ratio*100:.0f}%")
        print(f"{'='*70}")
        
        for epoch in range(1, num_epochs + 1):
            # Iterative pruning - gradually increase sparsity
            if (self.schedule == PruningSchedule.ITERATIVE and 
                iterative_prune_epochs and 
                epoch in iterative_prune_epochs and
                self.technique != PruningTechnique.NONE):
                
                prune_step = iterative_prune_epochs.index(epoch)
                current_ratio = self.prune_ratio * (prune_step + 1) / len(iterative_prune_epochs)
                print(f"\n[Epoch {epoch}] Applying pruning (ratio: {current_ratio:.2f})")
                self._apply_pruning(current_ratio)
            
            # One-shot after - prune in middle of training
            if (self.schedule == PruningSchedule.ONESHOT_AFTER and 
                epoch == num_epochs // 2 and
                self.technique != PruningTechnique.NONE):
                print(f"\n[Epoch {epoch}] Applying one-shot pruning, then fine-tuning...")
                self._apply_pruning(self.prune_ratio)
            
            # Training step
            result = self.algo.train()
            
            # Get model statistics
            sparsity_fc1, sparsity_fc2 = self.policy.model.get_layer_sparsity()
            
            # Prepare statistics
            stat = {
                "epoch": epoch,
                "exp_name": self.exp_name,
                "technique": self.technique.value,
                "schedule": self.schedule.value,
                "target_prune_ratio": self.prune_ratio,
                "actual_sparsity": self.policy.model.get_sparsity(),
                "fc1_sparsity": sparsity_fc1,
                "fc2_sparsity": sparsity_fc2,
                "reward_mean": result.get('episode_reward_mean', 0),
                "reward_max": result.get('episode_reward_max', 0),
                "reward_min": result.get('episode_reward_min', 0),
                "episode_len": result.get('episode_len_mean', 0),
            }
            
            # Log to file
            self._log(stat)
            self.stats.append(stat)
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "reward/mean": stat['reward_mean'],
                    "reward/max": stat['reward_max'],
                    "reward/min": stat['reward_min'],
                    "episode_len": stat['episode_len'],
                    "sparsity/total": stat['actual_sparsity'],
                    "sparsity/fc1": stat['fc1_sparsity'],
                    "sparsity/fc2": stat['fc2_sparsity'],
                })
            
            print(f"Epoch {epoch:2d} | "
                  f"Reward: {stat['reward_mean']:7.2f} | "
                  f"Max: {stat['reward_max']:7.2f} | "
                  f"Sparsity: {stat['actual_sparsity']*100:5.1f}%")
        
        self.logf.close()
        
        # Log final summary
        final_stats = self.get_final_stats()
        if self.use_wandb:
            wandb.summary.update({
                "final/avg_reward": final_stats['avg_reward'],
                "final/avg_max_reward": final_stats['avg_max_reward'],
                "final/sparsity": final_stats['final_sparsity'],
            })
            wandb.finish()
        
        return final_stats
    
    def get_final_stats(self):
        """Return average stats from last 5 epochs"""
        if len(self.stats) < 5:
            last_n = self.stats
        else:
            last_n = self.stats[-5:]
        
        return {
            "exp_name": self.exp_name,
            "technique": self.technique.value,
            "schedule": self.schedule.value,
            "prune_ratio": self.prune_ratio,
            "avg_reward": np.mean([s['reward_mean'] for s in last_n]),
            "avg_max_reward": np.mean([s['reward_max'] for s in last_n]),
            "final_sparsity": self.stats[-1]['actual_sparsity'],
        }

# ============================================================
# Grid Search Runner
# ============================================================
def run_comprehensive_grid_search(
    techniques=None,
    schedules=None,
    sparsity_levels=None,
    num_epochs=30,
    use_wandb=True,
    wandb_project="rl-pruning-grid-search"
):
    """
    Run comprehensive grid search over techniques, schedules, and sparsity levels.
    
    Args:
        techniques: List of PruningTechnique to test
        schedules: List of PruningSchedule to test
        sparsity_levels: List of sparsity ratios to test
        num_epochs: Number of epochs per experiment
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
    """
    
    # Default configurations
    if techniques is None:
        techniques = [
            PruningTechnique.NONE,
            PruningTechnique.RANDOM_WEIGHT,
            PruningTechnique.MAGNITUDE_WEIGHT,
            PruningTechnique.FISHER_WEIGHT,
            PruningTechnique.HESSIAN_WEIGHT,
            PruningTechnique.RANDOM_NEURON,
            PruningTechnique.MAGNITUDE_NEURON,
            PruningTechnique.FISHER_NEURON,
            PruningTechnique.HESSIAN_NEURON,
        ]
    
    if schedules is None:
        schedules = [
            PruningSchedule.ONESHOT_BEFORE,
            PruningSchedule.ONESHOT_AFTER,
            PruningSchedule.ITERATIVE,
        ]
    
    if sparsity_levels is None:
        sparsity_levels = [0.25, 0.50, 0.75]
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE PRUNING GRID SEARCH")
    print(f"{'='*70}")
    print(f"Techniques: {len(techniques)}")
    print(f"Schedules: {len(schedules)}")
    print(f"Sparsity Levels: {len(sparsity_levels)}")
    print(f"Total Experiments: {len(techniques) * len(schedules) * len(sparsity_levels) + 1}")
    print(f"{'='*70}\n")
    
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    results = []
    baseline_reward = None
    
    # Run baseline
    print("\n>>> Running BASELINE...")
    exp = PruningExperiment(
        technique=PruningTechnique.NONE,
        schedule=PruningSchedule.ONESHOT_BEFORE,
        prune_ratio=0.0,
        exp_name="baseline",
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )
    stats = exp.train(num_epochs=num_epochs)
    baseline_reward = stats['avg_reward']
    results.append(stats)
    
    # Run grid search
    exp_count = 1
    total_exps = len(techniques) * len(schedules) * len(sparsity_levels)
    
    for tech in techniques:
        if tech == PruningTechnique.NONE:
            continue
            
        for schedule in schedules:
            for sparsity in sparsity_levels:
                exp_count += 1
                
                # Create experiment name
                exp_name = f"{tech.value}_{schedule.value}_sparsity{int(sparsity*100)}"
                
                print(f"\n>>> Running [{exp_count}/{total_exps}]: {exp_name}")
                
                # Setup iterative epochs if needed
                iterative_epochs = None
                if schedule == PruningSchedule.ITERATIVE:
                    iterative_epochs = [
                        num_epochs // 4,
                        num_epochs // 2,
                        3 * num_epochs // 4,
                        num_epochs - 5
                    ]
                
                # Run experiment
                exp = PruningExperiment(
                    technique=tech,
                    schedule=schedule,
                    prune_ratio=sparsity,
                    exp_name=exp_name,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project
                )
                
                stats = exp.train(num_epochs=num_epochs, iterative_prune_epochs=iterative_epochs)
                
                # Calculate performance retention
                if baseline_reward is not None and baseline_reward > 0:
                    stats['performance_retention'] = stats['avg_reward'] / baseline_reward
                
                results.append(stats)
    
    ray.shutdown()
    
    # Print summary
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"Baseline Reward: {baseline_reward:.2f}")
    print(f"{'='*70}\n")
    
    # Sort by performance
    results_sorted = sorted(results, key=lambda x: x['avg_reward'], reverse=True)
    
    print(f"{'Rank':<6} {'Experiment':<50} {'Sparsity':<10} {'Reward':<10} {'Retention':<10}")
    print(f"{'-'*95}")
    
    for rank, stat in enumerate(results_sorted[:20], 1):  # Top 20
        retention = stat.get('performance_retention', 1.0) * 100
        print(f"{rank:<6} "
              f"{stat['exp_name']:<50} "
              f"{stat['final_sparsity']*100:<10.1f} "
              f"{stat['avg_reward']:<10.2f} "
              f"{retention:<10.1f}%")
    
    # Save results
    results_file = f"grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {results_file}")
    
    return results

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("NEURAL NETWORK PRUNING FOR RL POLICY INFERENCE")
    print("Part of: Accelerating RL via Faster Policy Inference")
    print("="*70)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nUsage: python pruning_experiments.py [mode]")
        print("\nAvailable modes:")
        print("  quick       - Quick test (fewer combinations)")
        print("  full        - Full grid search (all combinations)")
        print("  custom      - Custom configuration")
        print("\nRunning 'quick' by default...\n")
        mode = "quick"
    
    if mode == "full":
        # Full grid search: all techniques × all schedules × all sparsity levels
        run_comprehensive_grid_search(
            num_epochs=30,
            use_wandb=True,
            wandb_project="rl-pruning-full"
        )
        
    elif mode == "quick":
        # Quick test with key techniques
        run_comprehensive_grid_search(
            techniques=[
                PruningTechnique.NONE,
                PruningTechnique.MAGNITUDE_WEIGHT,
                PruningTechnique.MAGNITUDE_NEURON,
                PruningTechnique.FISHER_NEURON,
            ],
            schedules=[
                PruningSchedule.ONESHOT_BEFORE,
                PruningSchedule.ITERATIVE,
            ],
            sparsity_levels=[0.25, 0.50, 0.6, 0.75, 0.8, 0.85],
            num_epochs=25,
            use_wandb=True,
            wandb_project="rl-pruning-quick"
        )
        
    elif mode == "custom":
        # Custom experiment
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        
        exp = PruningExperiment(
            technique=PruningTechnique.FISHER_NEURON,
            schedule=PruningSchedule.ITERATIVE,
            prune_ratio=0.50,
            exp_name="custom_fisher_iterative_50",
            use_wandb=True,
            wandb_project="rl-pruning-custom"
        )
        
        exp.train(
            num_epochs=30,
            iterative_prune_epochs=[5, 10, 15, 20, 25]
        )
        
        ray.shutdown()
    else:
        print(f"Unknown mode: {mode}")