from PPO import PPOTrainer
from Models import Agent
import gymnasium as gym
from My_wrapper import RenderFrameWrapper_CarRacing

Net1 = [
                ("conv", 32, 5, 1),
                ("BatchNorm",),
                ("relu",),
                ("conv", 64, 5, 1),
                ("BatchNorm",),
                ("relu",),
                ("pool", "max", 2, 2),
                ("conv", 64, 5, 1),
                ("BatchNorm",),
                ("relu",),
                ("conv", 128, 5, 1),
                ("BatchNorm",),
                ("relu",),
                ("pool", "max", 2, 2),
                ("conv", 128, 5, 1),
                ("BatchNorm",),
                ("relu",),
                ("conv", 128, 5, 1),
                ("BatchNorm",),
                ("relu",),
                ("pool", "adaptive", 1)
            ]

Net3 = [
                ("conv", 32, 5, 1),
                ("GroupNorm",),
                ("relu",),
                ("conv", 64, 5, 1),
                ("GroupNorm",),
                ("relu",),
                ("pool", "max", 2, 2),
                ("conv", 64, 5, 1),
                ("GroupNorm",),
                ("relu",),
                ("conv", 128, 5, 1),
                ("GroupNorm",),
                ("relu",),
                ("pool", "max", 2, 2),
                ("conv", 128, 5, 1),
                ("GroupNorm",),
                ("relu",),
                ("conv", 128, 5, 1),
                ("GroupNorm",),
                ("relu",),
                ("pool", "adaptive", 1)
            ]

def Train_teacher():
    Teacher = Agent(
        envs=gym.make("CarRacing-v3", render_mode="rgb_array"),
        cnn_channels=Net3,
        actor_layers=[128, 128],
        critic_layers=[128, 128],
    )

    default_config = {
        "env_name": "CarRacing-v3",
        "seed": 1000,
        "torch_deterministic": True,
        "num_envs": 32,
        "num_steps": 512,
        "num_iterations": 2000,
        "learning_rate": 2.5e-4,
        "anneal_lr": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.001,
        "anneal_ent_coef": True,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 20,
        "minibatch_size": 256,
        "norm_adv": True,
        "target_reward": 900,
        'Test_frequency': 10,
        "reset_after_test": True,
        'Verbose_frequency': 1,
        'wrappers': [RenderFrameWrapper_CarRacing]
    }
    folder_path = "./Results/CarRacing/PPO/Teacher/"
    dict_enviroment = {
        "run_name": "PPO_CarRacing_Teacher_GroupNorm",
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [],
        "folder_path" : folder_path
    }
    dict_test_enviroment = {
        "run_name": dict_enviroment["run_name"],
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [0],
        "seeds": list(range(100, 110)),
        "folder_path" : folder_path
    }

    PPO_trainer = PPOTrainer(
        agent=Teacher,
        path_folder=dict_enviroment["folder_path"],
        dict_enviroment=dict_enviroment,
        device='cuda:4',
        config=default_config,
        dict_test_enviroment=dict_test_enviroment,
    )

    PPO_trainer.train()


if __name__ == "__main__":
    Train_teacher()