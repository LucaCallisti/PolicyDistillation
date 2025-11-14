import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb
import time
import os
from My_env import my_LunarLander
from Utils_PPO import set_seeds
from Test_model import test_model
import time


def make_env(dict, idx, wrappers = []):       
    def thunk():
        if (idx in dict["render_idx"]) or (idx in dict["record_video_idx"]):
            if dict['env_name'] == "LunarLander":
                env = my_LunarLander(render_mode='rgb_array')
            else:
                env = gym.make(dict['env_name'], render_mode='rgb_array')
            if idx in dict["record_video_idx"]:
                dict['video_folder'] = os.path.join(dict["folder_path"], "videos", dict["run_name"])
                env = gym.wrappers.RecordVideo(env,  dict['video_folder'], episode_trigger=lambda episode_id: True)
        else:
            if dict['env_name'] == "LunarLander":
                env = my_LunarLander()
            else:
                env = gym.make(dict['env_name'])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    return thunk

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
    ):
        self.path_folder = path_folder
        self.run_name = dict_enviroment["run_name"]
        self.env_name = dict_enviroment["env_name"]
        self.device = device
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
        self.config = config

        # --- Setup envs ---
        wrapper = dict_enviroment.get('wrappers', [])
        self.envs = gym.vector.AsyncVectorEnv([make_env(dict_enviroment, idx, wrappers=wrapper) for idx in range(config["num_envs"])])
        if dict_test_enviroment is not None:
            self.test_envs = gym.vector.AsyncVectorEnv([make_env(dict_test_enviroment, idx, wrappers=wrapper) for idx in range(len(dict_test_enviroment['seeds']))], autoreset_mode = 'Disabled')
            self.dict_test_enviroment = dict_test_enviroment

        # --- Select agent type ---
        self.agent = agent.to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["learning_rate"], eps=1e-5)

        # --- Initialize buffers ---
        obs_shape = self.agent.input_shape
        action_shape = self.envs.single_action_space.shape

        self.obs = torch.zeros((config["num_steps"], config["num_envs"]) + obs_shape, device=device)
        self.actions = torch.zeros((config["num_steps"], config["num_envs"]) + action_shape, device=device)
        self.logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.rewards = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.dones = torch.zeros((config["num_steps"], config["num_envs"]), device=device)
        self.values = torch.zeros((config["num_steps"], config["num_envs"]), device=device)

        # --- Environment state ---
        self.next_obs, self.info = self.envs.reset(seed=config["seed"])
        set_seeds(config['seed'][0]) if isinstance(config['seed'], list) else set_seeds(config['seed'])
        self.next_obs = torch.tensor(self.next_obs, device=device)
        self.next_done = torch.zeros(config["num_envs"], device=device)

        self.global_step = 0
        self.start_time = time.time()

        config_log = {**self.config, **dict_enviroment}
        if dict_test_enviroment is not None:
            config_log.update(dict_test_enviroment)
        wandb.init(
            project=f"PPO_{self.env_name}",
            name=self.run_name,
            config=config_log,
            dir=self.path_folder,
            save_code=True,
        )
 


    @torch.inference_mode()
    def collect_rollouts(self):
        cfg = self.config
        reward_ended_episodes = []
        running_average = -999

        for step in range(cfg["num_steps"]):
            self.dones[step] = self.next_done

            # Prepara input
            if self.agent.agent_type == "Image":
                screen = self.agent.get_screen(envs=None, screen=self.next_obs.cpu())
                input_ = self.agent.update_memory(screen, self.next_done.cpu()).to(self.device)
            else:
                input_ = self.next_obs.to(self.device)

            self.obs[step] = input_

            # Azione
            action, logprob, _, value = self.agent.get_action_and_value(input_)
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.values[step] = value.flatten()

            # Step ambiente
            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
            self.next_obs, self.next_done = torch.tensor(next_obs, device=self.device), torch.tensor(next_done, device=self.device)

            if next_done.sum() > 0:
                reward_ended_episodes.extend(infos['episode']['r'][next_done])

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

        num_samples = b_obs.shape[0]
        b_inds = np.arange(num_samples)
        clipfracs = []

        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Sanity check
                if start == 0 and epoch == 0:
                    if not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).all(): 
                        print(f"Warning: Ratio not close to 1 at first update! elements differ. {torch.abs(ratio - 1.0).max().item()} max diff.")
                        breakpoint()

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

        return {
            'training/pg_loss' : pg_loss.item(),
            'training/v_loss' : v_loss.item(),
            'training/entropy' : entropy_loss.item(),
            'training/approx_kl' : approx_kl.item(),
            'training/clipfrac' : np.mean(clipfracs),
            'training/grad_norm' : grad_norm.item(),
            'training/ent_coef' : curr_ent_coef,
        }

    def train(self):
        cfg = self.config
        n_better = 0

        start_time = time.time()
        print('Start training...')
        for self.iteration in range(1, cfg["num_iterations"] + 1):
            if cfg["anneal_lr"]:
                frac = 1.0 - (self.iteration - 1.0) / cfg["num_iterations"]
                self.optimizer.param_groups[0]["lr"] = frac * cfg["learning_rate"]

            avg_rew, reward_episodes = self.collect_rollouts()
            self.global_step += cfg["num_envs"] * cfg["num_steps"]
            
            with torch.inference_mode():
                if self.agent.agent_type == "Image":
                    screen = self.agent.get_screen(envs=None, screen=self.next_obs.cpu())
                    input_ = self.agent.update_memory(screen, self.next_done.cpu()).to(self.device)
                else:
                    input_ = self.next_obs.to(self.device)
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

            if n_better >= 100:
                print(f"Solved at iteration {self.iteration}: Avg reward = {avg_rew:.2f}")
                mean_rwd_test = self.test()
                info = {
                    "training/avg_reward": avg_rew,
                    "test/avg_reward": mean_rwd_test,
                    "iteration": self.iteration,
                }
                self.agent.save_model(path = self.path_folder, title = self.run_name + "_ppo", wandb_bool = True, info_dict = info)
                break
            
            if self.iteration % self.config['Verbose_frequency'] == 0:
                print(f"[{self.iteration}/{cfg['num_iterations']}] AvgRew: {avg_rew:.2f}")
            
            if self.iteration % self.config["Test_frequency"] == 0 or self.iteration == 1:
                if self.agent.agent_type == "Image":
                    stored_input = self.agent.get_input_manager()
                mean_rwd_test = self.test()
                if self.agent.agent_type == "Image":
                    self.agent.set_input_manager(stored_input)
                info = {
                    "training/avg_reward": avg_rew,
                    "test/avg_reward": mean_rwd_test,
                    "iteration": self.iteration,
                }
                if self.iteration != 1: self.agent.save_model(path = self.path_folder, title = self.run_name + f"it{self.iteration}_ppo", wandb_bool = True, info_dict = info)
                self.agent.train()
                self.agent.deterministic = False

            # Logging
            wandb.log({
                "training/avg_reward": avg_rew,
                **logs,
                "training/SPS": int(self.global_step / (time.time() - self.start_time)),
                'training/n_better_episodes': n_better,
                "iteration": self.iteration,
                'global_step': self.global_step
            }, step=self.global_step)


        print("Training completed.", time.time() - start_time)
        self.envs.close()


    def test(self):
        mean_reward = test_model(model_A=self.agent, env =self.test_envs, iteration=self.iteration, global_step = self.global_step, seeds=self.dict_test_enviroment['seeds'], video_folder=self.dict_test_enviroment['video_folder'], device = self.device)
        return mean_reward