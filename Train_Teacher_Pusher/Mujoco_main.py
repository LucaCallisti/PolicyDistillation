import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import gymnasium as gym
import wandb
import numpy as np
import time
import torch
import random
from typing import Dict, Any
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from callbacks import My_EvalCallback, VideoCallback


def set_seed(seed: int):
    """Set seed for reproducibility across all libraries"""
    print(f"🌱 Setting seed to: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SAC training for Pusher-v5")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--seed-agent", type=int, default=0, help="Random seed for the agent")
    parser.add_argument("--seed-env", type=int, default=50, help="Random seed for the environment")
    parser.add_argument("--seed-eval-env", type=int, default=100, help="Random seed for the environment")

    
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gradient-steps", type=int, default=4, help="Number of gradient steps per update (SAC)")
    parser.add_argument("--sde-sample-freq", type=int, default=4, help="SDE sample frequency")
    
    # Training config
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--num-enviroments", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--time-limit", type=int, default=100, help="Episode time limit")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--reward-threshold", type=float, default=-10, help="Reward threshold to stop training")
    
    # Device and logging
    parser.add_argument("--device", type=str, default="cuda:4", help="Device (auto, cuda:0, cuda:1, cpu)")
    parser.add_argument("--project", type=str, default="Pusher", help="W&B project name")
    
    # Test only
    parser.add_argument("--test-only", action="store_true", help="Only run test, don't train")
    parser.add_argument("--load-from", type=str, default=None, help="Path to load model from")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # --- Set seed for reproducibility ---
    set_seed(args.seed)
    set_random_seed(args.seed)

    # --- Weights & Biases configuration ---
    wandb.init(
        entity = 'Distillation_RL',
        project=args.project,
        name=f"SAC_Pusher",
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
    )
    
    config = wandb.config
    log_dir = "/home/l.callisti/Distillation_LunarLander/Final_pipeline/Results/Pusher/Teacher"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Log directory: {log_dir}")
    print(f"Config: {dict(config)}")
    
    if False:
        # --- Test only mode ---
        test_model(args, log_dir)
    else:
        # --- Training mode ---
        train_model(args, log_dir)
    
    wandb.finish()


def train_model(args, log_dir):
    """Train the SAC model"""
    config = wandb.config
    
    # --- Environment creation ---
    print("Creating environments...")
    # Wrap each training env with Monitor so episodic rewards/lengths are recorded
    env = DummyVecEnv([
        lambda: Monitor(gym.make(
            "Pusher-v5",
            max_episode_steps=config.time_limit
        )) for _ in range(config.num_enviroments)
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    env.seed(args.seed_env)

    eval_env = DummyVecEnv([
        lambda: RecordVideo(
                    Monitor(
                        gym.make(
                            "Pusher-v5",
                            render_mode="rgb_array",
                            max_episode_steps=config.time_limit,
                            default_camera_config={
                                "distance": 3.5,
                                "lookat": (0.0, 0.0, 0.0),
                                "azimuth": 270,
                                "elevation": -25,
                            }
                        ),
                    filename=None  # No file logging
                    ),
                    video_folder=os.path.join(log_dir, "eval_videos"),
                    episode_trigger=lambda x: x % (args.n_eval_episodes + 1) == 0,
                    name_prefix="eval"
            )
    ])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.seed(args.seed_eval_env)
    video_callback = VideoCallback(video_folder=os.path.join(log_dir, "eval_videos"), upload_freq=config.eval_freq)
    
    # --- PPO model definition ---
    print("Creating SAC model...")
    # device resolution: allow 'auto' to select GPU0 if available
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SAC(
        "MlpPolicy",       # policy fully connected
        env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,
        tau=0.005,
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=args.gradient_steps,
        ent_coef="auto" if args.ent_coef == 'auto' else args.ent_coef,
        device=device,
    )

    # --- Callback for evaluation and logging ---
    print("Setting up callbacks...")
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=config.reward_threshold,
        verbose=1
    )
    eval_callback = My_EvalCallback(
        eval_env=eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        seed = args.seed_eval_env,
        model = model,
        save_dir = os.path.join(log_dir, "model_folder")
    )

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.eval_freq,
        save_path=checkpoint_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=log_dir,
        verbose=2,
    )
        
    # --- Training ---
    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback, wandb_callback, video_callback, checkpoint_callback],
    )

    # --- Final save ---
    print("Saving model...")
    model.save(os.path.join(log_dir, "sac_pusher_final"))
    env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    
    print("Training completed!")


def test_model(args, log_dir):
    """Test the trained model and save videos"""
    import imageio
    
    if args.load_from is None:
        args.load_from = log_dir
    
    print(f"Loading model from: {args.load_from}")
    
    model_path = os.path.join(args.load_from, "sac_pusher_final")
    vecnorm_path = os.path.join(args.load_from, "vecnormalize.pkl")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found: {model_path}.zip")
        return
    
    if not os.path.exists(vecnorm_path):
        print(f"VecNormalize not found: {vecnorm_path}")
        return
    
    # --- Test environment creation ---
    test_env = DummyVecEnv([
        lambda: gym.make(
            "Pusher-v5",
            render_mode="rgb_array",
            default_camera_config={
                "distance": 3.5,
                "lookat": (0.0, 0.0, 0.0),
                "azimuth": 270,
                "elevation": -25,
            }
        )
    ])
    
    test_env = VecNormalize.load(vecnorm_path, test_env)
    test_env.training = False
    test_env.norm_reward = False
    
    model = SAC.load(model_path, env=test_env)
    
    # --- Salvataggio video ---
    video_dir = os.path.join(args.load_from, "test_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Saving videos to: {video_dir}")
    
    obs = test_env.reset()
    episode = 0
    frames = []
    
    for step in range(100000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        frame = test_env.render()
        if frame is not None:
            frame_array = frame[0]
            if frame_array.ndim == 3 and frame_array.shape[0] == 3:
                frame_array = np.transpose(frame_array, (1, 2, 0))
            frames.append(frame_array)
        
        if done[0]:
            if episode < 5:
                video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
                imageio.mimsave(
                    video_path,
                    frames,
                    fps=20,
                    macro_block_size=1
                )
                print(f"Video {episode} saved: {video_path}")
            frames = []
            episode += 1
            obs = test_env.reset()
            
            if episode >= 5:
                break
    
    test_env.close()
    # Upload saved videos to W&B
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    if not video_files:
        print("No videos found to upload to W&B.")
    else:
        videos = []
        for i, fname in enumerate(video_files):
            path = os.path.join(video_dir, fname)
            try:
                videos.append(wandb.Video(path, fps=30, caption=f"episode_{i}"))
            except Exception as e:
                print(f"Failed to prepare video {path} for upload: {e}")
        if videos:
            wandb.log({"test_videos": videos}, step=0)
            print(f"Uploaded {len(videos)} videos to W&B.")
    print("Test completed!")


if __name__ == "__main__":
    main()

