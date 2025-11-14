from PPO import PPOTrainer
from Models import Agent
import gymnasium as gym

def Train_teacher():
    Teacher = Agent(
        envs=gym.make("CarRacing-v3", render_mode="rgb_array"),
        cnn_channels= [
                ("conv", 32, 5, 1),
                ("conv", 64, 5, 1),
                ("pool", "max", 2, 2),
                ("conv", 64, 5, 1),
                ("conv", 128, 5, 1),
                ("pool", "max", 2, 2),
                ("conv", 128, 5, 1),
                ("conv", 128, 5, 1),
                ("pool", "adaptive", 1)
            ],
        actor_layers=[128, 128],
        critic_layers=[128, 128],
    )

    default_config = {
        "env_name": "CarRacing-v3",
        "seed": 1000,
        "torch_deterministic": True,
        "num_envs": 16,
        "num_steps": 1024,
        "num_iterations": 1000,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
        "minibatch_size": 128,
        "norm_adv": True,
        "anneal_lr": True,
        "anneal_ent_coef": True,
        "target_reward": 900,
        'Test_frequency': 25,
        "reset_after_test": True,
        'Verbose_frequency': 1,
    }
    folder_path = "./Results/CarRacing/PPO/Teacher/"
    dict_enviroment = {
        "run_name": "PPO_CarRacing_Teacher",
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [],
        "folder_path" : folder_path
    }
    dict_test_enviroment = {
        "run_name": "PPO_CarRacing_Teacher",
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
        device='cuda:7',
        config=default_config,
        dict_test_enviroment=dict_test_enviroment,
    )

    PPO_trainer.train()


if __name__ == "__main__":
    Train_teacher()