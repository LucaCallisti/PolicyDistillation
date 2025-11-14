from PPO import PPOTrainer
from Models import Agent
from My_env import my_LunarLander
import os


def Train_teacher():
    Teacher = Agent(
        envs=my_LunarLander(),
        actor_layers=[128, 128],
        critic_layers=[128, 128],
    )

    default_config = {
        "env_name": "LunarLander",
        "seed": 1000,
        "torch_deterministic": True,
        "num_envs": 16,
        "num_steps": 1024,
        "num_iterations": 1000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.3,
        "clip_vloss": True,
        "ent_coef": 0.001,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
        "minibatch_size": 128,
        "norm_adv": True,
        "anneal_lr": True,
        "anneal_ent_coef": True,
        "target_reward": 200,
        'Test_frequency': 25,
        "reset_after_test": False,
        'Verbose_frequency': 25,
    }
    folder_path = "./Results/LunarLander/PPO/Teacher/"
    dict_enviroment = {
        "run_name": "PPO_LunarLander_Teacher",
        "env_name": default_config["env_name"],
        "render_idx": [],
        "record_video_idx": [],
        "folder_path" : folder_path
    }
    dict_test_enviroment = {
        "run_name": "PPO_LunarLander_Teacher",
        "env_name": default_config["env_name"],
        "render_idx": [],
        "record_video_idx": [0],
        "seeds": list(range(100, 110)),
        "folder_path" : folder_path
    }

    PPO_trainer = PPOTrainer(
        agent=Teacher,
        path_folder=dict_enviroment["folder_path"],
        dict_enviroment=dict_enviroment,
        device='cuda:5',
        config=default_config,
        dict_test_enviroment=dict_test_enviroment,
    )

    PPO_trainer.train()

if __name__ == "__main__":
    Train_teacher()