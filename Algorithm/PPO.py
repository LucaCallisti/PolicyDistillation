import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from Algorithm.seeds import get_training_seed
import gymnasium as gym
import wandb
import time
from Enviroment.Utils import make_env
from Algorithm.Test_model import test_model
import time
from Algorithm.Utils import get_input, set_seeds


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

torch.autograd.set_detect_anomaly(True)
class PPOTrainer:
    def __init__(
        self,
        agent,
        path_folder,
        dict_enviroment,
        device,
        config,
        dict_test_enviroment = None,
        Async_env: bool = True,
        run_index : int = 0
    ):
        self.path_folder = path_folder
        self.run_name = dict_enviroment["run_name"] + f"_run{run_index}"
        self.env_name = dict_enviroment["env_name"]
        self.device = device
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
        self.config = config

        # --- Setup envs ---
        print("Setting up environments...")
        if Async_env:
            self.vec_env_fun = gym.vector.AsyncVectorEnv
        else:
            self.vec_env_fun = gym.vector.SyncVectorEnv

        wrapper = dict_enviroment.get('wrappers', [])
        self.envs = self.vec_env_fun([make_env(dict_enviroment, idx, wrappers=wrapper) for idx in range(config["num_envs"])])
        if dict_test_enviroment is not None:
            self.dict_test_enviroment = dict_test_enviroment

        # --- Select agent type ---
        print("Setting up agent and optimizer...")
        self.agent = agent.to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["learning_rate"], eps=1e-5)

        # --- Initialize buffers ---
        print("Initializing buffers...")
        obs_shape = self.agent.input_shape['State']
        action_shape = self.envs.single_action_space.shape

        self.obs = torch.zeros((config["num_steps"], config["num_envs"]) + obs_shape, device=device)
        self.actions = torch.zeros((config["num_steps"], config["num_envs"]) + action_shape, device=device)
        self.logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.rewards = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.dones = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.values = torch.zeros((config["num_steps"], config["num_envs"]), device=device)

        # --- Environment state ---
        print("Resetting environments...")
        self.next_obs, self.info = self.envs.reset(seed=get_training_seed(run_index))
        set_seeds(get_training_seed(run_index))
        self.next_obs = torch.tensor(self.next_obs, device=device)
        self.next_done = torch.zeros(config["num_envs"], device=device)

        self.global_step = 0
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
            project=self.env_name,
            name=f"PPO_{self.run_name}",
            config=config_log,
            dir=self.path_folder,
            save_code=True,
        )
        print(self.agent)
 
    @torch.inference_mode()
    def collect_rollouts(self):
        cfg = self.config
        reward_ended_episodes = []
        running_average = -999

        for step in range(cfg["num_steps"]):
            self.dones[step] = self.next_done

            
            input_ = get_input(self.agent, self.next_done, self.info)
            self.obs[step] = input_
            if hasattr(self.agent, 'obs_rms'):
                self.agent.update_obs_rms(input_.reshape(-1, *input_.shape[2:]))

            # Action
            action, logprob, _, value = self.agent.get_action_and_value(input_)
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.values[step] = value.flatten()

            # Environment step
            next_obs, reward, terminations, truncations, self.info = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
            self.next_obs, self.next_done = torch.tensor(next_obs, device=self.device), torch.tensor(next_done, device=self.device)

            if next_done.sum() > 0:
                reward_ended_episodes.extend(self.info['episode']['r'][next_done])

        avg_rew = float(np.mean(reward_ended_episodes)) if reward_ended_episodes else float('nan')
        return avg_rew, reward_ended_episodes


    def compute_gae(self, next_value):
        cfg = self.config
        advantages = torch.zeros_like(self.rewards, device=self.device)
        lastgaelam = 0

        for t in reversed(range(cfg["num_steps"])):
            if t == cfg["num_steps"] - 1:
                nextnonterminal = 1.0 - self.next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]

            delta = self.rewards[t] + cfg["gamma"] * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam

        returns = advantages + self.values
        return advantages, returns


    def ppo_update(self, advantages, returns):
        cfg = self.config

        # Flatten
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_new_values = torch.zeros_like(b_values)
        explained_var = torch.zeros(cfg["update_epochs"])

        num_samples = b_obs.shape[0]
        b_inds = np.arange(num_samples)
        clipfracs = []

        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                b_new_values[mb_inds] = newvalue.view(-1).detach()
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

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

                entropy_loss = entropy.mean()
                curr_ent_coef = cfg["ent_coef"] * (0.99 ** self.iteration if cfg["anneal_ent_coef"] else 1)
                loss = pg_loss - curr_ent_coef * entropy_loss + cfg["vf_coef"] * v_loss



                self.optimizer.zero_grad()
                loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

            explained_var[epoch] = 1 - (torch.var(b_returns - b_new_values) / torch.var(b_returns))

        return {
            'training/pg_loss' : pg_loss.item(),
            'training/v_loss' : v_loss.item(),
            'training/entropy' : entropy_loss.item(),
            'training/approx_kl' : approx_kl.item(),
            'training/clipfrac' : np.mean(clipfracs),
            'training/grad_norm' : grad_norm.item(),
            'training/ent_coef' : curr_ent_coef,
            'training/explained_variance' : explained_var.mean().item(),
        }

    def _save_metrics_df(self):
        """Save the metrics DataFrame to the run folder."""
        csv_path = os.path.join(self.run_folder, 'metrics.csv')
        self.df_metrics.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

    def train(self):
        cfg = self.config
        n_better = 0

        start_time = time.time()
        print('Start training...')
        for self.iteration in range(1, cfg["num_iterations"] + 1):
            print(f"Iteration {self.iteration}/{cfg['num_iterations']}", end='\r')
            if cfg["anneal_lr"]:
                frac = 1.0 - (self.iteration - 1.0) / cfg["num_iterations"]
                self.optimizer.param_groups[0]["lr"] = frac * cfg["learning_rate"]

            avg_rew, reward_episodes = self.collect_rollouts()
            self.global_step += cfg["num_envs"] * cfg["num_steps"]
            
            with torch.inference_mode():
                input_ = get_input(self.agent, self.next_done, self.info)
                next_value = self.agent.get_value(input_).reshape(1, -1)

            advantages, returns = self.compute_gae(next_value)
            logs = self.ppo_update(advantages, returns)

            rewards_np = np.array(reward_episodes)
            if (rewards_np > cfg["target_reward"]).all():
                n_better += len(reward_episodes)
            else:
                mask = rewards_np > cfg["target_reward"]
                false_idx = np.where(mask == False)[0]
                n_better = rewards_np.size - false_idx[-1] - 1

            # Logging
            log_data = {
                "training/avg_reward": avg_rew,
                **logs,
                "training/SPS": int(self.global_step / (time.time() - self.start_time)),
                'training/n_better_episodes': n_better,
                "training/learning_rate": self.optimizer.param_groups[0]["lr"],
                "iteration": self.iteration,
                'global_step': self.global_step
            } 
            wandb.log(log_data, step=self.global_step)
            
            # Append to DataFrame
            new_row = pd.DataFrame([log_data])
            self.df_metrics = pd.concat([self.df_metrics, new_row], ignore_index=True)

            if n_better >= 100:
                print(f"Solved at iteration {self.iteration}: Avg reward = {avg_rew:.2f}")
                mean_rwd_test = self.test()
                info = {
                    "training/avg_reward": avg_rew,
                    "test/avg_reward": mean_rwd_test,
                    "iteration": self.iteration,
                }
                self.agent.save_model(path=self.run_folder, title=self.run_name + "_ppo", wandb_bool=True, info_dict=info)
                self._save_metrics_df()
                break
            
            if self.iteration % self.config['Verbose_frequency'] == 0:
                print(f"[{self.iteration}/{cfg['num_iterations']}] AvgRew: {avg_rew:.2f}")
            
            if self.iteration % self.config["Test_frequency"] == 0 or self.iteration == 1 or self.iteration >= cfg["num_iterations"]:
                if "Frame" in self.agent.input_type:
                    stored_input = self.agent.get_input_manager()
                    self.agent.reset_input_manager()
                mean_rwd_test = self.test()
                if "Frame" in self.agent.input_type:
                    self.agent.set_input_manager(stored_input)
                info = {
                    "training/avg_reward": avg_rew,
                    "test/avg_reward": mean_rwd_test,
                    "iteration": self.iteration,
                }
                if self.iteration != 1:
                    self.agent.save_model(path=self.run_folder, title=self.run_name + f"it{self.iteration}_ppo", wandb_bool=True, info_dict=info)
                self.agent.train()
                self.agent.deterministic = False

                self.df_metrics = pd.concat([self.df_metrics, pd.DataFrame([self.info_test_OD])], ignore_index=True)
                
                # Save DataFrame at each test
                self._save_metrics_df()

        print("Training completed.", time.time() - start_time)
        self._save_metrics_df()
        self.envs.close()


    def test(self):
        test_envs = self.vec_env_fun([make_env(self.dict_test_enviroment, idx, wrappers=self.dict_test_enviroment['wrappers']) for idx in range(len(self.dict_test_enviroment['ODseeds']))], autoreset_mode = 'Disabled')
        mean_reward, self.info_test_OD = test_model(model_A=self.agent, env = test_envs, iteration=self.iteration, global_step = self.global_step, seeds=self.dict_test_enviroment['ODseeds'], video_folder=self.dict_test_enviroment['video_folder'], title = 'OD test')
        return mean_reward
    
