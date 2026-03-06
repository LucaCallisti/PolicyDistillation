import os
import pickle
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC
import json
import pickle
import torch.nn as nn

from pathlib import Path
import sys

parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

import Models

def save_policy_weights(model_path, vecnorm_path=None   , output_dir=None):
    """
    Extract and save only the policy network weights from a saved SAC model.
    
    Args:
        model_path (str): Path to the saved SAC model (without .zip extension)
        output_dir (str): Directory where to save weights. If None, creates 'policy_weights' 
                         in the same directory as the model
    """
    # Load the SAC model
    model = SAC.load(model_path)
    
    # Define output directory
    if output_dir is None:
        model_dir = os.path.dirname(model_path)
        output_dir = os.path.join(model_dir, "policy_weights")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting and saving policy weights...")
    
    # Save the policy weights (actor network)
    policy_weights_path = os.path.join(output_dir, "policy_weights.pth")
    torch.save(model.policy.actor.state_dict(), policy_weights_path)
    print(f"Policy weights saved to: {policy_weights_path}")
    
    layers_info = []
    for name, module in model.policy.actor.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers_info.append({
                "type": "Linear",
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
        elif isinstance(module, torch.nn.ReLU):
            layers_info.append({
                "type": "ReLU",
                "name": name,
            })
        elif isinstance(module, torch.nn.Tanh):
            layers_info.append({
                "type": "Tanh",
                "name": name,
            })
    
    # Save the architecture
    architecture_info = {
        "policy_type": "MlpPolicy",
        "input_dim": int(model.observation_space.shape[0]),
        "output_dim": int(model.action_space.shape[0]),
        "learning_rate": float(model.learning_rate) if hasattr(model.learning_rate, 'item') else float(model.learning_rate),
        "layers": layers_info,
    }
    arch_path = os.path.join(output_dir, "architecture.json")
    with open(arch_path, 'w') as f:
        json.dump(architecture_info, f, indent=2)
    print(f"Architecture info saved to: {arch_path}")
    
    # Save the critic weights (optional but useful for continual learning)
    critic_weights_path = os.path.join(output_dir, "critic_weights.pth")
    torch.save(model.critic.qf0.state_dict(), critic_weights_path)
    print(f"Critic-network weights saved to: {critic_weights_path}")

    if vecnorm_path and os.path.exists(vecnorm_path):
        # Load VecNormalize with pickle (without environment)
        with open(vecnorm_path, 'rb') as f:
            vecnorm = pickle.load(f)

        # Save statistics in JSON (human-readable)
        if hasattr(vecnorm, 'obs_rms'):
            vecnorm_stats = {
                "obs_mean": vecnorm.obs_rms.mean.tolist(),
                "obs_var": vecnorm.obs_rms.var.tolist(),
                "norm_obs": vecnorm.norm_obs,
                "norm_reward": vecnorm.norm_reward,
                "clip_obs": float(vecnorm.clip_obs),
                "clip_reward": float(vecnorm.clip_reward),
            }
            
            vecnorm_stats_path = os.path.join(output_dir, "vecnormalize_stats.json")
            with open(vecnorm_stats_path, 'w') as f:
                json.dump(vecnorm_stats, f, indent=2)
            print(f"VecNormalize stats saved to: {vecnorm_stats_path}")

    
    print(f"\n All weights and normalization extracted to: {output_dir}")
    return output_dir


def save_with_my_model(model_path, vecnorm_path=None, output_dir=None):
    actor_layers = [256, 256]
    envs = gym.make("Pusher-v5")
    my_model = Models.Agent(envs=envs, actor_layers=actor_layers, NonLinearity=nn.ReLU, NormalizationObs=True)

    sac_model = SAC.load(model_path)

    my_model.actor_backbone.load_state_dict(sac_model.policy.actor.latent_pi.state_dict())
    my_model.mu_layer.load_state_dict(sac_model.policy.actor.mu.state_dict())
    my_model.log_std_param.load_state_dict(sac_model.policy.actor.log_std.state_dict())

    with open(vecnorm_path, 'rb') as f:
            vecnorm = pickle.load(f)

    my_model.obs_rms.mean = torch.tensor(vecnorm.obs_rms.mean,  dtype=torch.float32)
    my_model.obs_rms.var = torch.tensor(vecnorm.obs_rms.var,  dtype=torch.float32)

    breakpoint()

    my_model.save_model(path=output_dir, title = 'TeacherModel', info_dict = {'source': 'Converted from SB3 SAC'})


if __name__ == "__main__":
    model_path = "/home/l.callisti/PolicyDistillation/Results/Pusher/SAC/checkpoints/sac_checkpoint_1600000_steps.zip"
    vecnorm_path = "/home/l.callisti/PolicyDistillation/Results/Pusher/SAC/checkpoints/sac_checkpoint_vecnormalize_1600000_steps.pkl"  
    output_dir = "/home/l.callisti/PolicyDistillation/Results/Pusher/Teacher"
    save_with_my_model(model_path, vecnorm_path, output_dir)