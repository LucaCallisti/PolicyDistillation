import os
import torch
import numpy as np
import gymnasium as gym
from Enviroment.Utils import make_env
from Algorithm.Utils import _get_max_steps, get_input, set_seeds, create_observation_dict
from PIL import Image
from Algorithm.seeds import DATASET_STARTING_SEED


# TODO: Add option for episodes: discard poor episodes, end episodes early
@torch.inference_mode()
def Collect_Dataset(
        model,
        num_data,
        path_folder,
        dict_enviroment,
        num_envs,
        additional_wrapper = [],
        additional_obs_shape = {},
        transform_frame = lambda x: x,
        device = 'cpu',
        enviroment = None,
    ):
    # --- Setup envs ---
    wrapper = dict_enviroment.get('wrappers', []) + additional_wrapper
    envs = gym.vector.AsyncVectorEnv([make_env(dict_enviroment, idx, wrappers=wrapper) for idx in range(num_envs)], autoreset_mode='Disabled')
    env_indices = np.arange(num_envs)

    # --- Initialize buffers ---    
    action_shape = envs.single_action_space.shape
    max_length_ep = _get_max_steps(envs.env_fns[0]())
    num_data_expanded = num_data + max_length_ep

    key_list = model.input_type if isinstance(model.input_type, list) else [model.input_type]
    key_list = key_list + list(additional_obs_shape.keys())
    obs_shape = model.input_shape
    obs_shape.update(additional_obs_shape)
        
    obs= create_observation_dict((max_length_ep, num_envs), obs_shape = obs_shape, key = key_list, device = device)
    actions = torch.zeros((max_length_ep, num_envs,) + action_shape, device = device)
    logprobs = torch.zeros((max_length_ep, num_envs), device = device)
    entropies = torch.zeros((max_length_ep, num_envs), device = device)
    if model.action_type == 'Continuous':
        logits = torch.zeros((max_length_ep, num_envs, 2 * action_shape[0]), device = device)
    elif model.action_type == 'Discrete':
        logits = torch.zeros((max_length_ep, num_envs, envs.single_action_space.n), device = device)
    rewards = torch.zeros((max_length_ep, num_envs))
    print('type of obs:', obs.keys())
    
    current_index = 0
    full_obs = create_observation_dict((num_data_expanded,), obs_shape = obs_shape, key = key_list)
    full_actions = torch.zeros((num_data_expanded,) + action_shape, dtype=torch.float32)
    full_logprobs = torch.zeros((num_data_expanded,), dtype=torch.float32)
    full_entropies = torch.zeros((num_data_expanded,), dtype=torch.float32)
    full_rewards = torch.zeros((num_data_expanded,), dtype=torch.float32)
    full_logits = torch.zeros((num_data_expanded,) + logits.shape[2:], dtype=torch.float32)
    full_seeds = torch.zeros((num_data_expanded,), dtype=torch.int64)
    new_episode = torch.zeros((num_data_expanded,), dtype=torch.bool)

    # --- Environment state ---
    set_seeds(DATASET_STARTING_SEED)
    def _create_array(raw, columns, starting_point):
        arr = np.arange(starting_point, starting_point + raw * columns).reshape(raw, columns)
        return arr
    seeds = _create_array(num_data_expanded // num_envs +1, num_envs, DATASET_STARTING_SEED)
    
    # --- Collect data ---
    model.eval() 
    model.deterministic = True
    model.to(device)

    i = -1
        
    while current_index < num_data:

        if enviroment == 'LunarLander':
            landed_episodes_counter = torch.ones(num_envs) * 5

        print('Collected data:', current_index, '/', num_data, end='\r')
        i += 1
        dones = np.full(num_envs, False)
        step = np.zeros(num_envs, dtype=int)
        state, info = envs.reset(seed=seeds[i].tolist()) 
        Image.fromarray(info['Frame'][0]).save(os.path.join(path_folder, "example_frame_no_preprocessed.png"))
        while not dones.all():
            input_ = get_input(model, next_done=dones, info=info)

            action, logprob, entropy = model.get_action(input_)
            logit = model.get_logits()

            active_envs = ~dones    

            for k in obs.keys():
                if k == 'Frame':
                    obs[k][step[active_envs], env_indices[active_envs]] = transform_frame(torch.as_tensor(info[k][active_envs], device=device)).squeeze()
                else:
                    obs[k][step[active_envs], env_indices[active_envs]] = torch.as_tensor(info[k][active_envs], device=device)

            if enviroment == 'LunarLander':
                left_leg = state[:, -2]
                right_leg = state[:, -1]
                landed_episodes = left_leg * right_leg
                landed_episodes_counter -= landed_episodes
                active_envs = ( (landed_episodes_counter > 0) & active_envs).bool()

            actions[step[active_envs], env_indices[active_envs]] = action[active_envs].float()
            logprobs[step[active_envs], env_indices[active_envs]] = logprob[active_envs]
            entropies[step[active_envs], env_indices[active_envs]] = entropy[active_envs]
            logits[step[active_envs], env_indices[active_envs]] = logit[active_envs]

            if isinstance(input_, tuple):
                for inp, k in zip(input_, obs.keys()):
                    obs[k][step[active_envs], env_indices[active_envs]] = inp[active_envs]
            else:
                key_list = list(obs.keys())
                obs[key_list[0]][step[active_envs], env_indices[active_envs]] = input_[active_envs]            

            state, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            rewards[step[active_envs], env_indices[active_envs]] = torch.tensor(reward[active_envs], dtype=torch.float32)

            if not isinstance(active_envs, np.ndarray):
                active_envs = active_envs.numpy()
            step = step + active_envs.astype(int)
            dones |= terminations | truncations
        
        # Store full data
        full_seeds[i * num_envs : i * num_envs + num_envs] = torch.tensor(seeds[i], dtype=torch.int64)
        for k in range(num_envs):
            length = step[k]
            new_episode[current_index] = True
            full_index_start = current_index
            full_index_end = current_index + length
            full_obs_slice = slice(full_index_start, full_index_end)
            step_slice = slice(0, length)

            for key in obs.keys():
                full_obs[key][full_obs_slice] = obs[key][step_slice, k].cpu()
            full_actions[full_obs_slice] = actions[step_slice, k].cpu()
            full_logprobs[full_obs_slice] = logprobs[step_slice, k].cpu()
            full_entropies[full_obs_slice] = entropies[step_slice, k].cpu()
            full_rewards[full_obs_slice] = rewards[step_slice, k].cpu()
            full_logits[full_obs_slice] = logits[step_slice, k].cpu()

            current_index += length
            if current_index >= num_data:
                break
    print('Collected data:', current_index, '/', num_data)

    # --- Save dataset ---
    dataset = {
        'num_data': current_index,
        'observations': full_obs,
        'actions': full_actions[:current_index],
        'logprobs': full_logprobs[:current_index],
        'logit_for_distribution': full_logits[:current_index],
        'entropies': full_entropies[:current_index],
        'rewards': full_rewards[:current_index],
        'new_episode': new_episode[:current_index],
        'seeds': full_seeds[:(i + 1) * num_envs],
    } 
    try:
        dataset['model'] = model._get_state_dict()
    except:
        pass

    # Save one example frame (if present) as PNG
    if 'Frame' in full_obs:
        from torchvision.utils import save_image
        frame = full_obs['Frame'][0]  # first saved frame

        # If channels-first (C,H,W) -> (H,W,C)
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            frame = frame.permute(1, 2, 0)
        # If single-channel stored as HxWx1 -> squeeze to HxW (grayscale)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame.squeeze(axis=2)
        save_image(frame, os.path.join(path_folder, "example_frame.png"))

    torch.save(dataset, os.path.join(path_folder, f"dataset_{current_index}steps.pt"))



    


