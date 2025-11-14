import os

import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


from PPO import PPOTrainer
from Models import Agent
import gymnasium as gym
from My_wrapper import Float32ObsWrapper

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


def Train_teacher(config=None):
    Teacher = Agent(
        envs=gym.make("Pusher-v5"),
        actor_layers=[128, 128],
        critic_layers=[128, 128],
    )

    default_config = {
        "env_name": "Pusher-v5",
        "seed": 1000,
        "torch_deterministic": True,
        "num_envs": 32,
        "num_steps": 256,
        "num_iterations": 3000,
        "learning_rate": 3e-4,
        "anneal_lr": True,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.001,
        "anneal_ent_coef": False,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 10,
        "minibatch_size": 64,
        "norm_adv": True,
        "target_reward": -10,
        'Test_frequency': 25,
        "reset_after_test": False,
        'Verbose_frequency': 1,
    }

    if config is not None:
        default_config.update(config)
    folder_path = "./Results/Pusher/PPO/Teacher/"
    dict_enviroment = {
        "run_name": "PPO_Pusher",
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [],
        "folder_path" : folder_path,
        'wrappers': [Float32ObsWrapper]
    }
    dict_test_enviroment = {
        "run_name": dict_enviroment["run_name"],
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [0],
        "seeds": list(range(100, 110)),
        "folder_path" : folder_path,
        'wrappers': [Float32ObsWrapper]
    }

    PPO_trainer = PPOTrainer(
        agent=Teacher,
        path_folder=dict_enviroment["folder_path"],
        dict_enviroment=dict_enviroment,
        device='cuda:0',
        config=default_config,
        dict_test_enviroment=dict_test_enviroment,
    )

    PPO_trainer.train()


if __name__ == "__main__":
    scheduler = ASHAScheduler(
        metric="reward_mean",
        mode="max",
        max_t=2000,         # Max iterazioni per trial
        grace_period=100,   # Aspetta 100 iterazioni prima di killare
        reduction_factor=2, # Scarta il 50% peggiore ogni volta
    )
    reporter = CLIReporter(
        metric_columns=[
            "reward_mean",
            "training/pg_loss",
            "training/entropy",
            "training_iteration"
        ]
    )

    # Definiamo lo spazio di ricerca
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "clip_coef": tune.uniform(0.1, 0.3),
        "ent_coef": tune.loguniform(1e-4, 1e-2),
        "vf_coef": tune.loguniform(0.1, 2.0),  # Value function coefficient
        "gae_lambda": tune.uniform(0.9, 0.99),  # GAE smoothing
        "update_epochs": tune.choice([5, 10, 20, 40]),  # Quante volte rielaborare dati
        "minibatch_size": tune.choice([64, 128, 256, 512]),  # Batch size
        "num_envs": tune.choice([8, 16, 32, 64]),  # Ambienti paralleli
        "hidden_size": tune.choice([64, 128, 256]),
        "gamma": tune.uniform(0.99, 0.999),  # Discount factor
    }
    result = tune.run(
        Train_teacher,
        config=search_space,
        scheduler=scheduler,
        num_samples=50,
        max_concurrent_trials=4,
        progress_reporter=reporter,
        resources_per_trial={"gpu": 0.25},  # ✅ 4 trial in parallelo (4 × 0.25 = 1 GPU)
        storage_path=os.path.abspath("ray_results/"),
        verbose=1,
    )
    # tuner = tune.Tuner(
    #     Train_teacher,
    #     param_space=search_space,
    #     tune_config=tune.TuneConfig(
    #         scheduler=scheduler,
    #         num_samples=50,
    #         max_concurrent_trials=4,  # Massimo 4 trial in parallelo
    #     ),
    #     run_config=air.RunConfig(
    #         name="Pusher_PPO_Search",
    #         storage_path=os.path.abspath("ray_results/"),  # ✅ CAMBIA da local_dir a storage_path
    #         progress_reporter=reporter,
    #         verbose=1,
    #     ),
    #     _remote_trial_exec_kwargs={"resources": {"gpu": 0.25}},
    # )
    # print("Starting hyperparameter search...")
    # results = tuner.fit()
    print("\n=== MIGLIORI 5 TRIAL ===")
    best_trials = results.get_best_result(metric="reward_mean", mode="max", n=5)
    for i, trial in enumerate(best_trials, 1):
        print(f"\n{i}. Config: {trial.config}")
        print(f"   Reward: {trial.last_result['reward_mean']:.2f}")
