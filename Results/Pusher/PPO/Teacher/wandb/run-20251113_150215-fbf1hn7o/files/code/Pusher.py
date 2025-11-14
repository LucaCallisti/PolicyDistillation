from PPO import PPOTrainer
from Models import Agent
import gymnasium as gym


def Train_teacher():
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
        "target_reward": -10,
        'Test_frequency': 10,
        "reset_after_test": True,
        'Verbose_frequency': 1,
    }
    folder_path = "./Results/Pusher/PPO/Teacher/"
    dict_enviroment = {
        "run_name": "PPO_Pusher",
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [],
        "folder_path" : folder_path,
        'wrappers': []
    }
    dict_test_enviroment = {
        "run_name": dict_enviroment["run_name"],
        "env_name": default_config["env_name"],
        "render_idx": list(range(0, default_config["num_envs"])),
        "record_video_idx": [0],
        "seeds": list(range(100, 110)),
        "folder_path" : folder_path,
        'wrappers': []
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