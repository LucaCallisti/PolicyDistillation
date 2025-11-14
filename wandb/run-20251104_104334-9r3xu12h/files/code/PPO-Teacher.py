# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from Utils_PPO import PPO_Agent, PPO_Image_model, get_screen, make_env, set_seeds
from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    device: str = "cuda:4" if torch.cuda.is_available() else "cpu"

    # Cambia questa riga
    enviroment: Literal["LunarLander", "CarRacing"] = "CarRacing"
    """the environment to train on"""
    

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4 #2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 1024 # 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    target_reward: float = 200
    """Average target reward that need to be reached for early stopping"""
    gamma: float = 0.99    # 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.98 # 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128 # 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    Anneal_ent_coef: bool = True
    """Anneal coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    path_folder: str = "Teacher_files"
    """the path to the folder where the model and other files are saved"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



def train():
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, args.capture_video, run_name, args.path_folder, args.enviroment) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.enviroment == 'LunarLander':
        agent = PPO_Agent(envs).to(args.device)
    elif args.enviroment == 'CarRacing':
        agent = PPO_Image_model(envs).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    if agent.agent_type == 'State':
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    else:
        obs, _ = envs.reset()
        input = agent.init_deque(get_screen(screen = obs))
        obs = torch.zeros((args.num_steps, args.num_envs) + input.shape[1:]).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    n_times_better_target_value = 0

    # Start the game
    global_step = 0
    start_time = time.time()
    set_seeds(args.seed, args.torch_deterministic)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)

    for iteration in range(1, args.num_iterations + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect episodes
        Running_Average = -999
        number_of_episodes = 0
        current_reward = torch.zeros(args.num_envs).to(args.device)
        Reward_ended_episodes = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            dones[step] = next_done

            if agent.agent_type == 'Image':
                if step == 0 and iteration == 1:
                    input = agent.init_deque(get_screen(envs = None, screen = next_obs.cpu()))
                else:
                    input = agent.append_deque(get_screen(envs = None, screen = next_obs.cpu()), next_done.cpu())
            else:
                input = next_obs
            obs[step] = input

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(input.to(args.device))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)

            current_reward += rewards[step]
            if next_done.sum() > 0:
                Running_Average = ( Running_Average * number_of_episodes + ( current_reward * next_done ).sum() ) / (number_of_episodes + next_done.sum())
            number_of_episodes += next_done.sum()
            Reward_ended_episodes += current_reward[next_done == 1].cpu().numpy().tolist()
            current_reward[next_done == 1] = 0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        wandb.log({"Teacher_training/charts/episodic_return": info["episode"]["r"]}, step=global_step)
                        wandb.log({"Teacher_training/charts/episodic_length": info["episode"]["l"]}, step=global_step)
        
        if Running_Average == -999:
            Running_Average = Reward_ended_episodes.mean()
        wandb.log({"Teacher_training/chart/Average_reward": Running_Average}, step=global_step)

        reward_arr = np.array(Reward_ended_episodes)
        if (reward_arr > args.target_reward).all():
            n_times_better_target_value += number_of_episodes
        else:
            mask = reward_arr > args.target_reward
            false_idx = np.where(mask == False)[0]
            n_times_better_target_value = reward_arr.size - false_idx[-1] - 1
        if n_times_better_target_value >= 100:
            print(f"Solved! Running_Average is {Running_Average}. Stopping training.")
            torch.save(agent.state_dict(), os.path.join(args.path_folder, f"{run_name}_ppo_agent.pt"))
            wandb.save(f"Teacher_files/{run_name}_ppo_agent.pt")
            break

        # Calculation GAE
        with torch.no_grad():
            breakpoint()
            next_value = agent.get_value(input).reshape(1, -1)   # bootstrap value if not done
            advantages = torch.zeros_like(rewards).to(args.device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)



        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Normalization advantage
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                if args.Anneal_ent_coef:
                    curr_ent_coef = args.ent_coef * 0.99**(iteration)
                else:
                    curr_ent_coef = args.ent_coef
                loss = pg_loss - curr_ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log({"Teacher_training/losses/Entropy_coef": curr_ent_coef}, step=global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        wandb.log({"Teacher_training/charts/learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)
        wandb.log({"Teacher_training/losses/value_loss": v_loss.item()}, step=global_step)
        wandb.log({"Teacher_training/losses/policy_loss": pg_loss.item()}, step=global_step)
        wandb.log({"Teacher_training/losses/entropy": entropy_loss.item()}, step=global_step)
        wandb.log({"Teacher_training/losses/old_approx_kl": old_approx_kl.item()}, step=global_step)
        wandb.log({"Teacher_training/losses/approx_kl": approx_kl.item()}, step=global_step)
        wandb.log({"Teacher_training/losses/clipfrac": np.mean(clipfracs)}, step=global_step)
        wandb.log({"Teacher_training/losses/explained_variance": explained_var}, step=global_step)
        wandb.log({"Teacher_training/charts/N times avg reward >= target": n_times_better_target_value}, step=global_step)
        print(f'SPS: {int(global_step / (time.time() - start_time))}, Avg rew: {Running_Average:.2f}, it: {iteration}/{args.num_iterations}, n times average reward >= {args.target_reward}: {n_times_better_target_value}/100')
        wandb.log({"Teacher_training/charts/SPS": int(global_step / (time.time() - start_time))}, step=global_step)

        # Upload videos to wandb e poi elimina i file locali
        if args.track and args.capture_video:
            import glob
            video_dir = os.path.join(args.path_folder, "videos", run_name)
            video_files = sorted(glob.glob(f"{video_dir}/*.mp4"))
            for video_path in video_files:
                wandb.log({"Teacher_training/video": wandb.Video(video_path, caption=os.path.basename(video_path), format="mp4"), "iteration": iteration})
                os.remove(video_path)

    # envs.close()
    for env in envs.envs:
        if hasattr(env, "close"):
            env.close()
    writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    print('batch size', args.batch_size, 'minibatch size', args.minibatch_size)

    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"Teacher_lr_{args.learning_rate:.2e}" 
    run_name += '_Anneal_lr' if args.anneal_lr else ''
    run_name += f'_ent_coef_{args.ent_coef:.2e}'
    run_name += '_Anneal_ent' if args.Anneal_ent_coef else ''

    if args.track:
        import wandb
        wandb.init(
            project=f"{args.enviroment}-Distillation",
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir=".")
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    args.path_folder = os.path.join('./'+args.enviroment, args.path_folder)
    if not os.path.exists(args.path_folder):
        os.makedirs(args.path_folder)
    train()

