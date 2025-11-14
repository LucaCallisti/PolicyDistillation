from My_env import my_LunarLander
from Utils_PPO import set_seeds, EpisodeVideoRecorder
import torch
import gymnasium as gym
import os
import wandb
import numpy as np
from Utils_PPO import Visual_input, get_screen

def _get_max_steps(tmp_env):
    while hasattr(tmp_env, "env"):
        if hasattr(tmp_env, "_max_episode_steps"):
            return tmp_env._max_episode_steps
        tmp_env = tmp_env.env
    print("Warning: Could not find _max_episode_steps, returning 2000 as default.")
    return 2000

def take_action(model, done, info):
    if model.agent_type == 'Image':
        screen = get_screen(screen = info['frame_observation'])
        input = model.update_memory(screen, dones = done)
        action, log_prob, entropy = model.get_action(input.to(next(model.parameters()).device))
    elif model.agent_type == 'State':
        obs_tensor = torch.tensor(info['state_observation'], dtype=torch.float32).to(next(model.parameters()).device)
        action, log_prob, entropy = model.get_action(obs_tensor)
    return action, log_prob, entropy


@torch.inference_mode()
def test_model(model_A, env, iteration, global_step, seeds, video_folder=None, loss_fun = None, model_B=None, title = '', title_A='Student', title_B='Teacher', device = 'cpu'):    
    max_length_ep = _get_max_steps(env.env_fns[0]())
    model_A.eval()
    model_A.deterministic = True
    num_episodes = len(seeds)
    
    Ep_rewards = np.zeros(num_episodes)
    All_entropies = torch.full((num_episodes, max_length_ep), np.nan)
    if loss_fun is not None and model_B is not None:
        log_prob_A = torch.zeros(num_episodes, max_length_ep)
        log_prob_B = torch.zeros(num_episodes, max_length_ep)

    if model_B is not None:
        model_B.eval()
        All_entropies_B = torch.full((num_episodes, max_length_ep), np.nan)
        Accuracy = np.full((num_episodes, max_length_ep), np.nan)

    obs, info = env.reset(seed = seeds)
   
    if model_A.agent_type == 'Image':
        init_screen = model_A.get_screen(screen = obs)
        state_frame = model_A.reset_memory(init_screen)
    if model_B is not None and model_B.agent_type == 'Image':
        init_screen_B = model_B.get_screen(screen = obs)
        state_frame_B = model_B.reset_memory(init_screen_B)

    done = np.full(num_episodes, False)
    t = 0
    while done.sum() < num_episodes:
        action, log_prob_A, entropy = take_action(model_A, done, info)
        All_entropies[:, t] = entropy
        action = action.cpu().numpy()

        if model_B is not None:
            action_B, log_prob_B, entropy_B = take_action(model_B, done, info)
            All_entropies_B[:, t] = entropy_B
            Accuracy[:, t] = (action == action_B.cpu().numpy())

        if loss_fun is not None and model_B is not None:
            log_prob_A[:, t] = log_prob_A
            log_prob_B[:, t] = log_prob_B            

        obs, reward, terminated, truncated, info = env.step(action)
        Ep_rewards[~done] = Ep_rewards[~done] + reward[~done]
        done |= terminated | truncated

        t += 1

    mean_reward = Ep_rewards.mean().item()
    wandb.log({f"Test/Episode Reward {title_A} on {title}": mean_reward, f"Test/Mean entropy {title_A} on {title}": np.mean(np.nanmean(All_entropies, axis = 1)), 'iteration': iteration, 'global_step': global_step})
    if model_B is not None:
        wandb.log({f"Test/Mean entropy {title_B} (under {title_A}) on {title}": np.mean(np.nanmean(All_entropies_B, axis = 1)), 'Test/Accuracy' : np.mean(np.nanmean(Accuracy, axis = 1)), 'iteration': iteration, 'global_step': global_step})
    if video_folder is not None:
        if os.path.exists(video_folder):
            for fname in sorted(os.listdir(video_folder)):
                if fname.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.gif')):
                    video_path = os.path.join(video_folder, fname)
                    try:
                        wandb.log(
                            {f"Test/Video {title_A} on {title}": wandb.Video(video_path, caption=fname, format="mp4"), 'iteration': iteration, 'global_step': global_step},
                        )
                    except Exception as e:
                        print(f"Warning: failed to upload video {video_path} to wandb: {e}")
                    os.remove(video_path)
        else:
            print(f"Warning: video folder '{video_folder}' not found.")
    
    print(f"Finished with reward: {np.round(mean_reward, 1)}, Reward per episode: {np.round(Ep_rewards, 1)}")
    return mean_reward


class TestModel_underTeacher:
    @torch.inference_mode()
    def __init__(self, path = None, Teacher = None, seeds = None, envs = None):

        if path is None and seeds is None:
            raise ValueError("Devi fornire o il percorso del set o i seed.")
        
        if path is not None:
            loaded_dict = torch.load(path)
            self.TeacherRuns = loaded_dict['TeacherRuns']
            self.seeds = loaded_dict['seeds']
        else:
            self.Teacher = Teacher
            self.seeds = seeds
            self.envs = envs
            self.TeacherRuns = {}

            device = next(Teacher.parameters()).device

            self.max_length_ep = _get_max_steps(envs.env_fns[0]())
            num_episodes = len(seeds)

            Ep_rewards = np.zeros(num_episodes)
            All_entropies = torch.full((self.max_length_ep, num_episodes), np.nan)
            log_prob_T = torch.zeros((self.max_length_ep, num_episodes))
            actions = torch.zeros((self.max_length_ep, num_episodes) + self.envs.single_action_space.shape)


            obs, info = envs.reset(seed=seeds)
            InputManager = Visual_input(init_screens=get_screen(screen = info['frame_observation']))

            obs_state = torch.zeros((self.max_length_ep, num_episodes)+info['state_observation'].shape[1:])
            frames = InputManager.get_input()
            obs_frame = torch.zeros((self.max_length_ep, num_episodes)+frames.shape[1:])

            set_seeds(seeds[0]) if isinstance(seeds, list) else set_seeds(seeds)
            done = np.full(num_episodes, False)

            times = np.zeros(num_episodes, dtype=int)
            t=0
            while done.sum() < num_episodes:
                action, log_prob, entropy = take_action(self.Teacher, done, info)

                obs_tensor = torch.tensor(info['state_observation'], dtype=torch.float32).to(next(self.Teacher.parameters()).device)
                frames = InputManager.update_deque(frames = get_screen(screen = info['frame_observation']), dones = done)

                All_entropies[t, :] = entropy
                log_prob_T[t, :] = log_prob
                obs_state[t, :] = obs_tensor
                obs_frame[t, :] = frames
                actions[t, :] = action
            
                action = action.cpu().numpy()

                obs, reward, terminated, truncated, info = envs.step(action)
                Ep_rewards[~done] = Ep_rewards[~done] + reward[~done]
                done |= terminated | truncated

                times = times + done.astype(int)
                t += 1
            
            for i, seed in enumerate(seeds):
                self.TeacherRuns[seed] = {'steps': times[i], 'reward': Ep_rewards[i], 'obs_state': obs_state[:times[i], i], 'obs_frame': obs_frame[:times[i], i], 'Actions': actions[:times[i], i], 'log_prob_T': log_prob_T[:times[i], i], 'entropies': All_entropies[:times[i], i]}
    
    @torch.inference_mode()
    def TestModel(self, model):
        accuracy_list = np.zeros(len(self.seeds)) 
        for i, seed in enumerate(self.seeds):
            if model.agent_type == 'Image':
                input = self.TeacherRuns[seed]['obs_frame'].to(next(model.parameters()).device)
            else:
                input = self.TeacherRuns[seed]['obs_state'].to(next(model.parameters()).device)
            Actions_Student = model(input)
            Actions_Teacher = self.TeacherRuns[seed]['Actions']
            accuracy_list[i] = (Actions_Student.cpu() == Actions_Teacher).float().mean()
        del input
        return np.mean(accuracy_list)

