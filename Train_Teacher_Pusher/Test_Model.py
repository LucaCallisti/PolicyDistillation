import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Configuration ---
LOG_DIR = "/home/l.callisti/Distillation_Mujoco/ppo_pusher_wandb/PPO_Pusher_gSDE_20251104-141912"  # Change to your run name
MODEL_NAME = "best_model"
NUM_EPISODES = 5
FPS = 20

# --- Check that the model exists ---
model_path = os.path.join(LOG_DIR, MODEL_NAME)
vecnorm_path = os.path.join(LOG_DIR, "vecnormalize.pkl")

if not os.path.exists(model_path + ".zip"):
    print(f"Error: model not found at {model_path}.zip")
    exit(1)

if not os.path.exists(vecnorm_path):
    print(f"Error: vecnormalize.pkl not found at {vecnorm_path}")
    exit(1)

print(f"Model found: {model_path}.zip")
print(f"VecNormalize found: {vecnorm_path}")

# --- Create test environment ---
test_env = DummyVecEnv([
    lambda: gym.make(
        "Pusher-v5",
        render_mode="rgb_array",
        max_episode_steps=100,
        default_camera_config={
            "distance": 3.5,
            "lookat": (0.0, 0.0, 0.0),
            "azimuth": 270,
            "elevation": -25,
        })
])

# Load normalization
test_env = VecNormalize.load(vecnorm_path, test_env)
test_env.training = False
test_env.norm_reward = False

# Load model
model = PPO.load(model_path, env=test_env)

# --- Save videos ---
video_dir = os.path.join(LOG_DIR, f"test_videos_camera")
os.makedirs(video_dir, exist_ok=True)

print(f"\nStarting test... saving {NUM_EPISODES} videos to: {video_dir}")

obs = test_env.reset()
episode = 0
frames = []
curr_step = 0

for step in range(1000000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    curr_step += 1
    print('done:', done, 'curr step:', curr_step)
    
    frame = test_env.render()
    if frame is not None:
        frames.append(frame)
    
    if done[0]:
        if episode < NUM_EPISODES:
            video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
            imageio.mimsave(
                video_path,
                frames,
                fps=FPS,
                macro_block_size=1
            )
            print(f"Video {episode} saved: {video_path}")
        frames = []
        episode += 1
        obs = test_env.reset()
        curr_step = 0
        
        if episode >= NUM_EPISODES:
            break

test_env.close()
print(f"\nTest completed! {NUM_EPISODES} videos saved.")
