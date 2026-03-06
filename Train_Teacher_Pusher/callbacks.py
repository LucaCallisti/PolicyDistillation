import numpy as np
import wandb
from typing import Any
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from typing import Optional, Union
import os

np.set_printoptions(precision=3, suppress=True) 

class My_EvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        seed = None,
        model = None,
        save_dir = None,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.initial_distance_array = np.zeros((len(self.eval_env.envs), self.n_eval_episodes))
        self.final_distance_array = np.zeros((len(self.eval_env.envs), self.n_eval_episodes))
        self.seed = seed
        self.has_vecnormalize = self._find_vecnormalize()
        self.model = model
        self.save_dir = save_dir

    def _find_vecnormalize(self):
        """Search for VecNormalize in the wrapper chain."""
        current = self.eval_env
        while hasattr(current, 'venv'):  # VecNormalize ha attributo 'venv'
            if isinstance(current, VecNormalize):
                return True
            current = current.venv
        return isinstance(current, VecNormalize)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)

        if locals_['current_lengths'][0] == 1:
            for i in range(len(self.eval_env.envs)) if hasattr(self.eval_env, 'envs') else [1]:
                if self.has_vecnormalize:
                    self.initial_obs = self.eval_env.get_original_obs()[i]
                else:
                    self.initial_obs = locals_["observations"][i]
                self.initial_distance = np.linalg.norm(self.initial_obs[-6:-3] - self.initial_obs[-3:])

        if locals_["done"]:
            for i in range(len(self.eval_env.envs)) if hasattr(self.eval_env, 'envs') else [1]:
                final_obs = self.old_observations[i]
                final_distance = np.linalg.norm(final_obs[-6:-3] - final_obs[-3:])

                assert (final_obs[-3:] == self.initial_obs[-3:]).all(), "Final goals do not match!"

                self.initial_distance_array[i][locals_['episode_counts'][0]] = self.initial_distance
                self.final_distance_array[i][locals_['episode_counts'][0]] = final_distance
        
            if locals_['episode_counts'][0] == self.n_eval_episodes-1:
                wandb.log({
                    "eval/mean_initial_distance": np.mean(self.initial_distance_array),
                    "eval/mean_final_distance": np.mean(self.final_distance_array),
                    'global_step': self.num_timesteps,
                })
                self.initial_distance_array = np.zeros((len(self.eval_env.envs), self.n_eval_episodes))
                self.final_distance_array = np.zeros((len(self.eval_env.envs), self.n_eval_episodes))
                if self.seed is not None:
                    self.eval_env.seed(seed=self.seed)
                if self.model is not None and self.save_dir is not None:
                    save_path = os.path.join(self.save_dir, f"model_eval_step_{self.num_timesteps}")
                    self.model.save(save_path)
        
        if self.has_vecnormalize:
            self.old_observations = self.eval_env.get_original_obs()
        else:
            self.old_observations = locals_["observations"]

class VideoCallback(BaseCallback):
    def __init__(self, video_folder, upload_freq=50000):
        super().__init__()
        self.video_folder = video_folder
        self.upload_freq = upload_freq
        self.last_uploaded = 0
    def _on_step(self) -> bool:
        # Upload videos every upload_freq timesteps
        if self.num_timesteps - self.last_uploaded >= self.upload_freq:
            self.last_uploaded = self.num_timesteps
            self._upload_videos()
        return True
    def _upload_videos(self):
        """Upload all new videos to W&B"""
        if not os.path.exists(self.video_folder):
            return
        
        video_files = sorted([
            f for f in os.listdir(self.video_folder) 
            if f.endswith(".mp4")
        ])
        
        if not video_files:
            return
                
        videos = []
        for fname in video_files:
            path = os.path.join(self.video_folder, fname)
            try:
                videos.append(wandb.Video(path, caption=fname, format="mp4"))
            except Exception as e:
                print(f"❌ Failed to prepare video {fname}: {e}")
        
        if videos:
            wandb.log({
                "eval/videos": videos,
                "global_step": self.num_timesteps
            })
        for fname in video_files:
            path = os.path.join(self.video_folder, fname)
            os.remove(path)


