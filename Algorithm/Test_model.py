
from Algorithm.Utils import set_seeds
from Enviroment.Utils import make_env
import torch
import gymnasium as gym
import os
import wandb
import numpy as np
from Algorithm.Utils import get_input, create_observation_dict, _get_max_steps, take_action, get_accuracy


@torch.inference_mode()
def test_model(model_A, env, iteration, global_step, seeds, video_folder=None, loss_fn = None, model_B=None, title = '', title_A='Student', title_B='Teacher', wandb_bool = True, wnadb_rec_bool = True):    
    if not isinstance(env, gym.vector.VectorEnv):
        max_length_ep = _get_max_steps(env)
    else:
        max_length_ep = _get_max_steps(env.env_fns[0]())
    
    model_A.eval()
    model_A.deterministic = True
    num_episodes = len(seeds)
    
    Ep_rewards = np.zeros(num_episodes)
    All_entropies = torch.full((num_episodes, max_length_ep), np.nan)

    device = next(model_A.parameters()).device

    if model_B is not None:
        model_B.eval()
        All_entropies_B = torch.full((num_episodes, max_length_ep), np.nan)
        Accuracy = np.full((num_episodes, max_length_ep), np.nan)

        if loss_fn is not None:
            total_loss = 0
            elements = 0
    
    obs, info = env.reset(seed = seeds)
    done = np.full(num_episodes, False)
    t = 0

    while done.sum() < num_episodes:
        print(f"Test episode step {t}/{max_length_ep}", end='\r')

        action, log_prob_A, entropy = take_action(model_A, done, info)

        All_entropies[:, t] = entropy
        action = action.cpu().numpy()

        if model_B is not None:
            action_B, log_prob_B, entropy_B = take_action(model_B, done, info)
            All_entropies_B[:, t] = entropy_B
            Accuracy[:, t] = get_accuracy(action, action_B, env)
            

        if loss_fn is not None and model_B is not None:
            active_envs = ~done
            loss_this_step = loss_fn(actions_S = torch.as_tensor(action).to(device), actions_T = torch.as_tensor(action_B).to(device), Student = model_A, Teacher = model_B, logits_T = None, reduction = 'none')[active_envs]
            total_loss += loss_this_step.sum()            
            elements += active_envs.sum()
            

        obs, reward, terminated, truncated, info = env.step(action)
        Ep_rewards[~done] = Ep_rewards[~done] + reward[~done]
        done |= terminated | truncated

        t += 1

    env.close()

    mean_reward = Ep_rewards.mean().item()
    if loss_fn is not None and model_B is not None:
        loss = total_loss / elements
    else:
        loss = None
    
    info = {
        f"Test/Episode Reward {title_A} on {title}": mean_reward,
        f"Test/Mean entropy {title_A} on {title}": np.mean(np.nanmean(All_entropies, axis = 1)),
    }
    if model_B is not None:
        info[f'Test/Accuracy (under {title_A}) on {title}'] = np.mean(np.nanmean(Accuracy, axis = 1))
        info[f"Test/Mean entropy {title_B} (under {title_A}) on {title}"] = np.mean(np.nanmean(All_entropies_B, axis = 1))
        if loss is not None:
            info[f"Test/Mean loss {title_A} vs {title_B} on {title}"] = loss

    if wandb_bool:
        info['iteration'] = iteration
        info['global_step'] = global_step
        wandb.log(info)

    if model_B is not None:
        accuracy_during_episode = np.nanmean(Accuracy, axis = 0) 
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(accuracy_during_episode)
        ax.set_ylim(0, 1)
        ax.set_title(f'iteration {iteration} - global step {global_step}')
        wandb.log({
            f"Test/{title} -accuracy_plot": wandb.Image(fig),
            "epoch" : iteration,
            "global_step": global_step
        })

    
    if wnadb_rec_bool:
        if video_folder is not None:
            if os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
            if os.listdir(video_folder) == []:
                print(f"No videos found in {video_folder} to upload.")
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
            print("No video folder provided, skipping video upload.")
    
    print(f"Finished with reward: {np.round(mean_reward, 1)}, Reward per episode: {np.round(Ep_rewards, 1)}, Accuracy: {np.round(np.mean(np.nanmean(Accuracy, axis = 1)) if model_B is not None else float('nan'), 4)}")
    # if wandb_bool:
    #     return mean_reward
    # else:
    #     return mean_reward, info
    return mean_reward, info


class TestModel_underTeacher:
    @torch.inference_mode()
    def __init__(self, path = None, Teacher = None, Student = None, seeds = None, envs = None):

        if path is None and seeds is None:
            raise ValueError("You must provide either a path to the set or the seeds.")
        
        if path is not None:
            loaded_dict = torch.load(path)
            self.TeacherRuns = loaded_dict['TeacherRuns']
            self.seeds = loaded_dict['seeds']
        else:
            self.Teacher = Teacher
            self.Teacher.deterministic = True
            self.Teacher.eval()
            self.max_length_ep = _get_max_steps(envs.env_fns[0]())
            num_episodes = len(seeds)
            obs_Teacher = create_observation_dict((self.max_length_ep, num_episodes), model = Teacher, device=next(Teacher.parameters()).device)
            obs_Student = create_observation_dict((self.max_length_ep, num_episodes), model = Student, device=next(Teacher.parameters()).device)
            
            self.seeds = seeds
            self.envs = envs
            self.TeacherRuns = {}

            Ep_rewards = np.zeros(num_episodes)
            All_entropies = torch.full((self.max_length_ep, num_episodes), np.nan)
            log_prob_T = torch.zeros((self.max_length_ep, num_episodes))
            actions = torch.zeros((self.max_length_ep, num_episodes) + self.envs.single_action_space.shape)

            _, info = envs.reset(seed=seeds)
            done = np.full(num_episodes, False)

            set_seeds(seeds[0]) if isinstance(seeds, list) else set_seeds(seeds)

            times = np.ones(num_episodes, dtype=int)  # ones to count the last step
            t=0
            while done.sum() < num_episodes:

                def _aux_update_obs(model, input_, obs_dict):
                    if not isinstance(input_, tuple):
                        obs_dict[model.input_type[0]][t] = input_
                    else:
                        for input, k in zip(input_, obs_dict.keys()):
                            obs_dict[k][t] = input

                input_T = get_input(Teacher, done, info)
                _aux_update_obs(Teacher, input_T, obs_Teacher)
                input_S= get_input(Student, done, info)
                _aux_update_obs(Student, input_S, obs_Student)
                
                action, log_prob, entropy = self.Teacher.get_action(input_T)
                
                All_entropies[t, :] = entropy
                log_prob_T[t, :] = log_prob
                actions[t, :] = action
            
                action = action.cpu().numpy()

                obs, reward, terminated, truncated, info = envs.step(action)
                Ep_rewards[~done] = Ep_rewards[~done] + reward[~done]
                done |= terminated | truncated

                times = times +  (1 - done.astype(int) )
                t += 1
            
            for i, seed in enumerate(seeds):
                aux = {k : obs_Student[k][:times[i], i] for k in obs_Student.keys()}
                self.TeacherRuns[seed] = {'steps': times[i], 'reward': Ep_rewards[i], 'observation': aux, 'Actions': actions[:times[i], i], 'log_prob_T': log_prob_T[:times[i], i], 'entropies': All_entropies[:times[i], i]}
            print("Teacher runs collected and stored in TeacherRuns dictionary.")
            for i, seed in enumerate(seeds):
                print(f"Seed: {seed}, Reward: {Ep_rewards[i]}")
    @torch.inference_mode()
    def TestModel(self, model):
        accuracy_list = np.zeros(len(self.seeds)) 
        device = next(model.parameters()).device
        for i, seed in enumerate(self.seeds):
            observation_ep = self.TeacherRuns[seed]['observation']
            if len(model.input_type) == 1:
                input_ = torch.as_tensor(observation_ep[model.input_type[0]], dtype=torch.float32).to(device)
            else:
                input_ = tuple([torch.as_tensor(observation_ep[k]).to(device) for k in model.input_type])
            
            Actions_Student = model(input_)

            Actions_Teacher = self.TeacherRuns[seed]['Actions']
            accuracy_list[i] = get_accuracy(Actions_Student, Actions_Teacher, self.envs)
        del input_
        return np.mean(accuracy_list)