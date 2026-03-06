import os
import time
import wandb
import pandas as pd
from Algorithm.PPO import PPOTrainer
from Algorithm.Test_model import test_model
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from Algorithm.Utils import get_input, load_to_observatoin_dict, set_seeds, create_observation_dict
from Enviroment.Utils import make_env
from Algorithm.seeds import get_training_seed


default_config = {
    "env_name": "LunarLander-v2",
    "seed": 42,
    "torch_deterministic": True,
    "num_envs": 8,
    "num_steps": 128,
    "num_iterations": 1000,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "update_epochs": 4,
    "minibatch_size": 256,
    "norm_adv": True,
    "anneal_lr": True,
    "anneal_ent_coef": False,
    "target_reward": 200,
}

class PPD(PPOTrainer):
    def __init__(self, Student: object, Teacher: object, path_folder: str, dict_enviroment: dict, device: str, config: dict, dict_test_enviroment: dict, Async_env: bool = True, run_index : int = 0):
        self.Teacher = Teacher.to(device)
        self.path_folder = path_folder
        self.run_name = dict_enviroment["run_name"] + f"_run{run_index}"
        self.env_name = dict_enviroment["env_name"]
        self.device = device
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
        self.config = config
        if Async_env:
            self.vec_env_fun = gym.vector.AsyncVectorEnv
        else:
            self.vec_env_fun = gym.vector.SyncVectorEnv

        wrapper = dict_enviroment.get('wrappers', [])
        self.envs = self.vec_env_fun([make_env(dict_enviroment, idx, wrappers=wrapper) for idx in range(config["num_envs"])])
        if dict_test_enviroment is not None:
            self.dict_test_enviroment = dict_test_enviroment

        print("Setting up agent and optimizer...")
        self.agent = Student.to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config["learning_rate"])

        state, self.info = self.envs.reset(seed=get_training_seed(run_index))
        set_seeds(get_training_seed(run_index))
        
        self.obs_Teacher = create_observation_dict((config["num_steps"], config["num_envs"]), model = Teacher, device=device)
        self.obs_Student = create_observation_dict((config["num_steps"], config["num_envs"]), model = Student, device=device)
        self.key_student = Student.input_type
        self.key_teacher = Teacher.input_type

        action_shape = self.envs.single_action_space.shape
        self.actions = torch.zeros((config["num_steps"], config["num_envs"]) + action_shape, device=device)
        self.logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.rewards = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.dones = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.values = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.next_obs = { k: torch.tensor(self.info[k], device=device) for k in self.key_student}
        self.next_done = torch.zeros(config["num_envs"], device=device)
        # self.logits_S = torch.zeros((config["num_steps"], config["num_envs"]) + (2 * action_shape[0],), device=device)      # Debug

        self.global_step = 0
        self.optimization_steps = 0  # number of gradient updates
        self.samples_seen = 0  # samples used for training (minibatch_size × updates)
        self.samples_collected = 0  # samples collected from rollouts
        self.start_time = time.time()
        
        # Create folder for this run
        self.run_folder = os.path.join(path_folder, self.run_name)
        os.makedirs(self.run_folder, exist_ok=True)
        
        # DataFrame to store all metrics
        self.df_metrics = pd.DataFrame()

        config_log = {**self.config, **dict_enviroment}
        if dict_test_enviroment is not None:
            config_log.update(dict_test_enviroment)

        wandb.init(
            entity = 'Distillation_RL',
            project=f"{self.env_name}",
            name=f"PPD_{self.run_name}",
            config=config_log,
            dir=self.path_folder,
            save_code=True,
        )

    @torch.inference_mode()
    def collect_rollouts(self):
        cfg = self.config
        reward_ended_episodes = []

        for step in range(cfg["num_steps"]):
            self.dones[step] = self.next_done

            # Prepare input
            input_Student = get_input(self.agent, self.next_done, self.info)
            load_to_observatoin_dict(self.obs_Student, input_Student, step)

            ######### PPD - Get Teacher observation
            input_Teacher = get_input(self.Teacher, self.next_done, self.info)
            load_to_observatoin_dict(self.obs_Teacher, input_Teacher, step)
            #########

            # Action
            action, logprob, _, value = self.agent.get_action_and_value(input_Student)
    
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.values[step] = value.squeeze()

            # Environment step
            _, reward, terminations, truncations, self.info = self.envs.step(action.detach().cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
            self.next_done = torch.tensor(next_done, device=self.device)
            self.next_obs = { k: torch.tensor(self.info[k], device=self.device) for k in self.key_student}

            if next_done.sum() > 0:
                reward_ended_episodes.extend(self.info['episode']['r'][next_done])

        # Update samples_collected (samples from rollouts)
        self.samples_collected += cfg["num_steps"] * cfg["num_envs"]
        
        avg_rew = float(np.mean(reward_ended_episodes)) if reward_ended_episodes else float('nan')
        return avg_rew, reward_ended_episodes
    
    def ppo_update(self, advantages, returns):
        cfg = self.config

        # Flatten
        b_obs_Student = { k: self.obs_Student[k].reshape((-1,) + self.obs_Student[k].shape[2:]) for k in self.key_student}
        b_obs_Teacher = { k: self.obs_Teacher[k].reshape((-1,) + self.obs_Teacher[k].shape[2:]) for k in self.key_teacher}
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # b_logits_S = self.logits_S.reshape((-1,) + self.logits_S.shape[2:])      # Debug

        ############
        with torch.no_grad():
            input_ = tuple(b_obs_Teacher[k] for k in self.key_teacher) if len(self.key_teacher) > 1 else b_obs_Teacher[self.key_teacher[0]]
            _ = self.Teacher.get_action(input_)
        b_logits_teacher = self.Teacher.get_logits()
        ############

        num_samples = b_actions.shape[0]
        b_inds = np.arange(num_samples)
        clipfracs = []

        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                input_ = tuple(b_obs_Student[k][mb_inds] for k in self.key_student) if len(self.key_student) > 1 else b_obs_Student[self.key_student[0]][mb_inds]
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(input_, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Sanity check
                if start == 0 and epoch == 0:
                    if not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).all(): 
                        print(f"Warning: Ratio not close to 1 at first update! {(not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).sum().item())} elements differ. {torch.abs(ratio - 1.0).max().item()} max diff.")
                        # if self.agent.action_type == 'Continuous':
                        #     old_logits = b_logits_S[mb_inds].detach().cpu().numpy()
                        #     new_logits = self.agent.get_logits().detach().cpu().numpy()
                        #     print('difference in logits:', np.abs(old_logits - new_logits).max())

                        # first_new_logits = new_logits[0]
                        # print('ratio indagato:', ratio[0].item())
                        # for i in range(b_logits_S.shape[0]): 
                        #     if (b_logits_S[i].cpu().numpy() == first_new_logits).all(): 
                        #         print("Found matching input in stored dict at index:", i, ' while new index is', mb_inds[0].item())
                        #         break
                           
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if cfg["norm_adv"]:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                ).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -cfg["clip_coef"], cfg["clip_coef"])
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                ###################### PPD Loss
                Teacher_distribution = self.Teacher.create_distribution_from_logits(b_logits_teacher[mb_inds])
                Student_distribution = self.agent.get_last_distribution()
                max_ratio = torch.max(ratio, torch.tensor(1 - cfg["clip_coef"], device=self.device))
                kl = torch.distributions.kl.kl_divergence(Teacher_distribution, Student_distribution)
                if kl.ndim == 2:
                    kl = kl.sum(dim=-1)
                PPD_loss = ( kl * max_ratio ).mean()
                ########################

                entropy_loss = entropy.mean()
                curr_ent_coef = cfg["ent_coef"] * (0.99 ** self.iteration if cfg["anneal_ent_coef"] else 1)
                loss = pg_loss - curr_ent_coef * entropy_loss + cfg["vf_coef"] * v_loss + cfg['PPD_coef'] * PPD_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                
                self.optimization_steps += 1
                self.samples_seen += len(mb_inds)
             

        return {
            # X-axis metrics
            'x_axis/optimization_steps': self.optimization_steps,
            'x_axis/samples_seen': self.samples_seen,
            'x_axis/samples_collected': self.samples_collected,
            'x_axis/epochs': self.iteration,
            # Training metrics
            'training/pg_loss' : pg_loss.item(),
            'training/v_loss' : v_loss.item(),
            'training/entropy' : entropy_loss.item(),
            'training/approx_kl' : approx_kl.item(),
            'training/clipfrac' : np.mean(clipfracs),
            'training/grad_norm' : grad_norm.item(),
            'training/ent_coef' : curr_ent_coef,
            'training/PPD_loss' : PPD_loss.item(),
        }
    

