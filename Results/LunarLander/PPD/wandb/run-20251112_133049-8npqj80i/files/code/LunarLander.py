from PPO import PPOTrainer
from Models import Agent
from My_env import my_LunarLander
import os
import gymnasium as gym
from PPD import PPD
from My_wrapper import RenderFrameWrapper


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


def Train_student_PPD():
    Teacher = Agent.load_model_from_path(path = "/home/l.callisti/Distillation_LunarLander/Final_pipeline/LunarLander/ppo_agent_PPO_LunarLander_Teacher_ppo.pth", env=my_LunarLander(), info = True)
    Net2 = [
                ("conv", 32, 5, 1),
                ("relu",),
                ("conv", 64, 5, 1),
                ("relu",),
                ("pool", "max", 2, 2),
                ("conv", 64, 5, 1),
                ("relu",),
                ("conv", 128, 5, 1),
                ("relu",),
                ("pool", "max", 2, 2),
                ("conv", 128, 5, 1),
                ("relu",),
                ("conv", 128, 5, 1),
                ("relu",),
                ("pool", "adaptive", 1)
            ]

    Student = Agent(
        envs=RenderFrameWrapper(my_LunarLander(render_mode="rgb_array")),
        cnn_channels=Net2,
        actor_layers=[128, 128],
        critic_layers=[128, 128],
    )
    default_config = {
        "env_name": "LunarLander",
        "seed": 100_000,
        "torch_deterministic": True,
        "num_envs": 18,         # PPD change
        "num_steps": 64,        # PPD change
        "num_iterations": 1737, # PPD change   (2 million frames / (18 envs * 64 steps) = 1736.11)
        "learning_rate": 3e-4,  # PPD change
        "gamma": 0.999,         # PPD change
        "gae_lambda": 0.9,      # PPD change
        "clip_coef": 0.3,
        "clip_vloss": True,
        "ent_coef": 0.0,        # PPD change
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
        "minibatch_size": 512,  # PPD change
        "norm_adv": True,
        "anneal_lr": True,
        "anneal_ent_coef": True,
        "target_reward": 200,
        'Test_frequency': 25,
        "reset_after_test": False,
        'Verbose_frequency': 25,
        'PPD_coef': 1.0,        # PPD change # Da provare (0.5, 1, 2, 5)
    }
    folder_path = "./Results/LunarLander/PPD/"
    dict_enviroment = {
        "run_name": f"PPD_LunarLander_PPDparam{default_config['PPD_coef']}",
        "env_name": default_config["env_name"],
        "render_idx": [list(range(0, default_config["num_envs"]))],
        "record_video_idx": [],
        "folder_path" : folder_path
    }
    dict_test_enviroment = {
        "run_name": dict_enviroment["run_name"],
        "env_name": default_config["env_name"],
        "render_idx": [list(range(0, default_config["num_envs"]))],
        "record_video_idx": [0],
        "seeds": list(range(100, 110)),
        "folder_path" : folder_path
    }
    PPD_trainer = PPD(
        Student=Student,
        Teacher=Teacher,
        path_folder=dict_enviroment["folder_path"],
        dict_enviroment=dict_enviroment,
        device='cuda:5',
        config=default_config,
        dict_test_enviroment=dict_test_enviroment,
    )
    PPD_trainer.train()

if __name__ == "__main__":
    Train_student_PPD()