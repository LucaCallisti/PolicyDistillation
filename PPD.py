from PPO import PPOTrainer, make_env
import torch
import numpy as np
from My_wrapper import RenderFrameWrapper
import torch.nn as nn
import gymnasium as gym
from Utils_PPO import set_seeds
import torch
from torch.distributions import kl_divergence, Distribution


class PPD(PPOTrainer):
    def __init__(self, Student: object, Teacher: object, path_folder: str, dict_enviroment: dict, device: str, config: dict, dict_test_enviroment: dict):
        super().__init__(agent=Student, path_folder=path_folder, dict_enviroment=dict_enviroment, device=device, config=config, dict_test_enviroment=dict_test_enviroment)
        self.Teacher = Teacher.to(device)

        self.envs = gym.vector.AsyncVectorEnv([make_env(dict_enviroment, idx, wrappers=[RenderFrameWrapper]) for idx in range(config["num_envs"])])
        if dict_test_enviroment is not None:
            self.test_envs = gym.vector.AsyncVectorEnv([make_env(dict_test_enviroment, idx, wrappers=[RenderFrameWrapper]) for idx in range(len(dict_test_enviroment['seeds']))], autoreset_mode = 'Disabled')
            self.dict_test_enviroment = dict_test_enviroment

        self.next_obs, self.info = self.envs.reset(seed=config["seed"])
        set_seeds(config['seed'][0]) if isinstance(config['seed'], list) else set_seeds(config['seed'])
        self.next_obs = torch.tensor(self.next_obs, device=device)
        self.next_done = torch.zeros(config["num_envs"], device=device)


        self.obs_Teacher = torch.zeros((config["num_steps"], config["num_envs"]) + self.Teacher.input_shape, device=self.device)
        

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

            ######### PPD - Get Teacher observation
            self.obs_Teacher[step] = torch.tensor(self.info["state_observation"]).to(self.device)
            #########

            # Azione
            action, logprob, _, value = self.agent.get_action_and_value(input_)
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.values[step] = value.flatten()

            # Step ambiente
            next_obs, reward, terminations, truncations, self.infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
            self.next_obs, self.next_done = torch.tensor(next_obs, device=self.device), torch.tensor(next_done, device=self.device)

            if next_done.sum() > 0:
                reward_ended_episodes.extend(self.infos['episode']['r'][next_done])

        avg_rew = float(np.mean(reward_ended_episodes)) if reward_ended_episodes else float('nan')
        return avg_rew, reward_ended_episodes
    
    def ppo_update(self, advantages, returns):
        cfg = self.config

        # Flatten
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        ############
        b_obs_Teacher = self.obs_Teacher.reshape((-1,) + self.obs_Teacher.shape[2:])
        with torch.no_grad():
            _ = self.Teacher.get_action(b_obs_Teacher)
        b_logits_teacher = self.Teacher.get_last_logits()
        ############

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
                        print(f"Warning: Ratio not close to 1 at first update! {(not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).sum().item())} elements differ. {torch.abs(ratio - 1.0).max().item()} max diff.")
                    

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
                PPD_loss = (torch.distributions.kl.kl_divergence(Teacher_distribution, Student_distribution) * torch.clamp(ratio, 1 - cfg["clip_coef"])).mean()
                ########################

                entropy_loss = entropy.mean()
                curr_ent_coef = cfg["ent_coef"] * (0.99 ** self.iteration if cfg["anneal_ent_coef"] else 1)
                loss = pg_loss - curr_ent_coef * entropy_loss + cfg["vf_coef"] * v_loss + cfg['PPD_coef'] * PPD_loss

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
            'training/PPD_loss' : PPD_loss.item(),
        }
    
