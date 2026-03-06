import torch
import numpy as np
import random
import os
import gymnasium as gym
import torchvision
from Algorithm.seeds import get_dataloader_seed
from torch.utils.data import Sampler


def get_input(model, next_done, info):
    device = next(model.parameters()).device
    if "Frame" in model.input_type:
        screen = model.get_screen(envs=None, screen=info['Frame'])
        if isinstance(next_done, torch.Tensor):
            next_done = next_done.cpu()
        input_frame = model.update_memory(screen, next_done).to(device)

    # if there is only one type of input, return it directly else return a tuple
    if len(model.input_type) == 1:
        if "Frame" in model.input_type:
            input_ = input_frame
        else:
            input_ = torch.tensor(info[model.input_type[0]], dtype=torch.float32).to(device)
    else:
        input_ = tuple([torch.as_tensor(info[k]).to(device) if k != "Frame" else input_frame for k in model.input_type])
    return input_

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)  # Extra determinism
    try:
        gym.utils.seeding.np_random(seed)
    except:
        pass    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA determinism


def create_observation_dict(init_shape, model = None, obs_shape = None, key = None, device = 'cpu', config = None):
    if model is not None:
        obs_shape = model.input_shape
        key = model.input_type
    elif obs_shape is None or key is None:
        raise ValueError("Either model or both obs_shape and key must be provided.")
    if isinstance(obs_shape, dict):
        obs = { k: torch.zeros(init_shape + obs_shape[k], device=device) for k in key}
    else:
        obs = { k: torch.zeros(init_shape + obs_shape, device=device) for k in key} 
    return obs
def load_to_observatoin_dict(obs_dict, input, step):
    if isinstance(input, tuple):
        if len(list(obs_dict.keys())) != len(input):
            raise ValueError("Length of input tuple does not match number of keys in obs_dict.")
        for inp, k in zip(input, obs_dict.keys()):
            obs_dict[k][step] = inp
    else:
        key_list = list(obs_dict.keys())
        if len(key_list) != 1:
            raise ValueError("obs_dict has multiple keys but input is not a tuple.")
        obs_dict[key_list[0]][step] = input
  

def _get_max_steps(tmp_env):
    while hasattr(tmp_env, "env"):
        if hasattr(tmp_env, "_max_episode_steps"):
            return tmp_env._max_episode_steps
        tmp_env = tmp_env.env
    print("Warning: Could not find _max_episode_steps, returning 2000 as default.")
    return 2000

def take_action(model, done, info):
    input_ = get_input(model, done, info)
    action, log_prob, entropy = model.get_action(x = input_)
    return action, log_prob, entropy

def get_accuracy(action1, action2, env):
    if isinstance(action2, torch.Tensor):
        action2 = action2.detach().cpu().numpy()
    if isinstance(action1, torch.Tensor):
        action1 = action1.detach().cpu().numpy()
    if isinstance(env.single_action_space, gym.spaces.Discrete):
        accuracy = (action1 == action2).mean().item()
    else:
        accuracy = 1 - ( (action1 - action2)**2  ).mean()
    return accuracy


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return idx
    
class CreateDataloaderFromDataset():
    def __init__(self, data_dict, alpha = 0.0, beta = 0.4, percentage_old_dataset=0.4, run_index=0, mode_alpha = 'constant', mode_sampling = 'classic'):
        # 1. Data initialization
        self.observations_original = data_dict['observations']  # Dictionary di tensori
        self.actions = data_dict['actions']
        self.logits_distribution = data_dict['logit_for_distribution']
        self.new_episode_flags = data_dict['new_episode']

        # 2. PER (Prioritized Experience Replay) parameters
        self.alpha = alpha
        self.initial_beta = beta
        self.beta = beta

        # 3. Buffer management (Old vs New Dataset)
        self.dataset_size = self.actions.shape[0]
        self.percentage_old_dataset = percentage_old_dataset
        self.split_index = int(self.dataset_size * self.percentage_old_dataset)
        self.write_index = self.split_index  # Write cursor
        self.new_samples_count = 0           # Total count of new samples seen
        self.samples_in_buffer = 0           # How many valid new samples we have now
        self.is_buffer_full = False          # Whether we have filled the allocated space

        # 4. Weight initialization
        self.weights = torch.ones(self.actions.shape[0])
        if alpha > 0.0:
            self.errors = torch.ones(self.actions.shape[0])
            self.exponential_error = torch.ones(self.actions.shape[0])

        # 5. Gaussian noise initialization
        self.noise_transform = None

        # 6. Set generator for reproducibility in DataLoader
        self.generator = torch.Generator()
        self.generator.manual_seed(get_dataloader_seed(run_index=run_index))

        # Test phase 2
        assert mode_alpha in ['linear', 'constant', 'dynamic_mean', 'dynamic_max'], "mode_alpha must be one of 'linear', 'constant', or 'dynamic'."
        self.mode_alpha = mode_alpha
        assert mode_sampling in ['classic', 'calssic_uniform', 'max', 'max_uniform'], "mode_sampling must be one of 'classic', 'classic_uniform', 'max', or 'max_uniform'."
        self.mode_sampling = mode_sampling

    def __len__(self):
        return self.actions.shape[0]

    def PreprocessFrames(self, num_frames, skipped_frames):
        # If num_frames = 3, skipped_frames = 2 then the indices will be
        # [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 4], [0, 2, 5], [0, 3, 6], [1, 4, 7]]
        
        if 'Frame' not in self.observations_original:
            raise ValueError("The key 'Frame' is not present in the observations.")

        # Compute episode sizes
        new_episode_index = torch.nonzero(self.new_episode_flags)
        episode_ends = torch.cat([new_episode_index[1:].squeeze(), torch.tensor([self.new_episode_flags.shape[0]])])
        episode_starts = new_episode_index.squeeze()
        episode_lengths = episode_ends - episode_starts
        max_ep_len = episode_lengths.max().item()

        current_frame = self.observations_original['Frame']  # Shape: [N, H, W]
        buffer_tensor = torch.zeros_like(current_frame).unsqueeze(1).repeat(1, num_frames, 1, 1)  # Shape: [N, num_frames, H, W]

        # Pre-compute all frame indices for the longest episode
        all_frame_indices = []
        for end_idx in range(max_ep_len):
            frame_indices = []
            for i in range(num_frames):
                target_idx = end_idx - (num_frames - 1 - i) * (skipped_frames + 1)
                if target_idx < 0:
                    frame_indices.append(0)
                else:
                    frame_indices.append(target_idx)
            all_frame_indices.append(frame_indices)

        current_seq_idx = 0  # Current sequence index
        
        # Process each episode, filling pre-allocated tensors directly
        for ep_start, ep_end in zip(episode_starts, episode_ends):
            ep_frames = current_frame[ep_start:ep_end]
            ep_len = len(ep_frames)
            indices_tensor = torch.tensor(all_frame_indices[:ep_len]) 
            selected_frames = ep_frames[indices_tensor]  # Shape: [num_valid, num_frames, H, W]
            buffer_tensor[current_seq_idx:current_seq_idx + ep_len] = selected_frames    
            current_seq_idx += ep_len

        self.observation = self.observations_original.copy()
        self.observation['Frame'] = buffer_tensor         
    
    def _update_priorities(self):
        if self.alpha <= 0.0:
            custom_sampler = torch.utils.data.RandomSampler(self.actions, generator=self.generator)
            self.weights = torch.ones(len(self.actions))
            return custom_sampler
        
        current_errors = self.errors
        self.priorities = (current_errors + 1e-6) ** self.alpha
        sampling_probs = self.priorities / self.priorities.sum()
        N = len(self.priorities)
        weights = (N * sampling_probs) ** (-self.beta)
        self.weights = weights / weights.max()
        custom_sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.priorities,
            num_samples=len(self.priorities),
            replacement=True,
            generator=self.generator
        )
        return custom_sampler
    
    def update_errors(self, idxs, new_errors):
        self.errors[idxs] = new_errors.cpu().clone()

    def _collate(self, batch_indices):
        idxs = torch.tensor(batch_indices, dtype=torch.long)
        actions_batch = self.actions[idxs]
        logits_distribution_batch = self.logits_distribution[idxs]
        obs_batch = {}
        for k in self.observation.keys():
            if k == 'Frame' and self.noise_transform is not None:
                obs_batch[k] = self.noise_transform(self.observation[k][idxs])
            else:
                obs_batch[k] = self.observation[k][idxs]
        weights_batch = self.weights[idxs] if self.alpha > 0.0 else torch.ones(len(idxs))
        masks_old, masks_new = self.get_old_new_sample_masks(idxs)
        return obs_batch, actions_batch, logits_distribution_batch, weights_batch, idxs, masks_old, masks_new
    
    def get_old_new_sample_masks(self, idxs):
        idxs_old = idxs < self.split_index
        if self.is_buffer_full:
            return idxs_old, ~idxs_old
        else:
            idxs_old = idxs_old | (idxs >= self.write_index)
            return idxs_old, ~idxs_old

    def GetDataloader(self, batch_size=32, gaussian_noise_std=0.0):
        """Create a DataLoader for pre-allocated tensors, with optional Gaussian noise augmentation."""

        # 1. Setup Augmentation
        if gaussian_noise_std > 0.0:
            if 'Frame' not in self.observations:
                raise ValueError("Gaussian noise requires 'Frame' in the observations.")
            self.noise_transform = torchvision.transforms.v2.GaussianNoise(sigma=gaussian_noise_std)
        else:
            self.noise_transform = None
        
        dataset = IndexDataset(len(self.actions))
        custom_sampler = self._update_priorities()     

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=custom_sampler,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=self._collate,
            generator=self.generator,
            worker_init_fn=seed_worker_fn
        )
        return dataloader

    def GetSequentialDataloader(self, batch_size=32, gaussian_noise_std=0.0):
        """Create a DataLoader that returns data in sequential order."""
        if gaussian_noise_std > 0.0:
            if 'Frame' not in self.observations:
                raise ValueError("Gaussian noise requires 'Frame' in the observations.")
            self.noise_transform = torchvision.transforms.v2.GaussianNoise(sigma=gaussian_noise_std)
        else:
            self.noise_transform = None

        dataset = IndexDataset(len(self.actions))
        sequential_sampler = torch.utils.data.SequentialSampler(self.actions)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sequential_sampler,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=self._collate,
            generator=self.generator,
            worker_init_fn=seed_worker_fn
        )
        return dataloader
    
    def get_balanced_datalaoder(self, batch_size=32, gaussian_noise_std=0.0):
        """Create two separate DataLoaders for old and new data."""
        if gaussian_noise_std > 0.0:
            if 'Frame' not in self.observations:
                raise ValueError("Gaussian noise requires 'Frame' in the observations.")
            self.noise_transform = torchvision.transforms.v2.GaussianNoise(sigma=gaussian_noise_std)
        else:
            self.noise_transform = None

        class BalancedBatchSampler(Sampler):
            def __init__(self, old_indices, new_indices, batch_size):
                self.old_indices = old_indices
                self.new_indices = new_indices
                self.batch_size = batch_size
                self.half_batch = batch_size // 2
                self.num_batches = max(len(old_indices), len(new_indices)) // self.half_batch

            def __iter__(self):
                old_idx_shuffled = self.old_indices.copy()
                new_idx_shuffled = self.new_indices.copy()
                random.shuffle(old_idx_shuffled)
                random.shuffle(new_idx_shuffled)
                
                old_ptr = 0
                new_ptr = 0
                
                # Build batches one by one
                for _ in range(self.num_batches):
                    batch = []
                    
                    # 1. Draw exactly half the batch from OLD data
                    for _ in range(self.half_batch):
                        if old_ptr >= len(old_idx_shuffled):
                            # If we ran out of old indices, reshuffle and restart
                            random.shuffle(old_idx_shuffled)
                            old_ptr = 0
                        batch.append(old_idx_shuffled[old_ptr])
                        old_ptr += 1
                        
                    # 2. Draw exactly half the batch from NEW data
                    for _ in range(self.half_batch):
                        if new_ptr >= len(new_idx_shuffled):
                            # If we ran out of new indices, reshuffle and restart
                            random.shuffle(new_idx_shuffled)
                            new_ptr = 0
                        batch.append(new_idx_shuffled[new_ptr])
                        new_ptr += 1
                        
                    # Optional but recommended: shuffle the final batch
                    # so the network doesn't always see old data first, then new
                    random.shuffle(batch)
                    
                    # "yield" returns the list of indices to the DataLoader and pauses the function
                    yield batch

            def __len__(self):
                # Return the total number of batches per epoch
                return self.num_batches

        dataset = IndexDataset(len(self.actions))
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=BalancedBatchSampler(
                old_indices=list(range(self.dataset_size)),
                new_indices=list(range(self.dataset_size, len(self.actions))),
                batch_size=batch_size
            ),
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=self._collate,
        )
        
        return dataloader

    def update_dataset(self, new_observations, new_actions, new_logits_distribution, loss, step_done, total_steps):
        # num_new_samples = new_actions.shape[0]
        # self.new_samples_count += num_new_samples
        
        # if num_new_samples > self.dataset_size-self.split_index:
        #     raise ValueError("the number of new samples exceeds the available space in the dataset. Consider increasing the dataset size or reducing the percentage of old dataset.")

        # if self.write_index + num_new_samples < self.dataset_size:
        #     selected_index = torch.arange(self.write_index, self.write_index + num_new_samples)
        #     self.write_index += num_new_samples
        #     if self.is_buffer_full:
        #         self.samples_in_buffer = self.dataset_size - self.split_index
        #     else: 
        #         self.samples_in_buffer += num_new_samples
        # else:
        #     self.is_buffer_full = True
        #     selected_index_1 = torch.arange(self.write_index, self.dataset_size)
        #     self.write_index = self.split_index
        #     self.samples_in_buffer = self.dataset_size - self.split_index
        #     selected_index_2 = torch.arange(self.write_index, self.write_index + (num_new_samples - selected_index_1.shape[0]))
        #     selected_index = torch.cat([selected_index_1, selected_index_2], dim=0)
        # for k in new_observations.keys():
        #     self.observation[k][selected_index] = new_observations[k].cpu().clone()
        # self.logits_distribution[selected_index] = new_logits_distribution.cpu().clone()
        # self.actions[selected_index] = new_actions.cpu().clone()
        # self.errors[selected_index] = loss.cpu().clone()

        for k in new_observations.keys():
            self.observation[k] = torch.cat((self.observation[k], new_observations[k].cpu().clone()))
        self.logits_distribution = torch.cat((self.logits_distribution, new_logits_distribution.cpu().clone()))
        self.actions = torch.cat((self.actions, new_actions.cpu().clone()))

        if self.alpha > 0.0:
            self._annealing_alpha(step_done, total_steps)

    def annealing_beta(self, current_iteration, total_iterations):
        self.beta = min(1.0, self.initial_beta + (1.0 - self.initial_beta) * (current_iteration / total_iterations) + 0.4)
    
    def _annealing_alpha(self, step_done, total_steps):
        if self.mode_alpha == 'linear':
            if not self.is_buffer_full:
                self.alpha = 0.2 + 0.4 * (step_done / (total_steps + 1e-6)) 
            else:
                self.alpha = 0.6  # If buffer is full, set alpha to constant value
        elif self.mode_alpha == 'constant':
            self.alpha = 0.6
        elif 'dynamic' in self.mode_alpha:
            max_oversampling_ratio = 3
            if self.is_buffer_full:
                if 'mean' in self.mode_alpha:
                    loss_new_samples = self.errors[self.split_index:].mean().item()
                else:
                    loss_new_samples = self.errors[self.split_index:].max().item()
                mean_loss_old_samples = self.errors[:self.split_index].mean().item()
            else:
                if 'mean' in self.mode_alpha:
                    loss_new_samples = self.errors[self.split_index:self.write_index].mean().item()
                else:
                    loss_new_samples = self.errors[self.split_index:self.write_index].max().item()
                mean_loss_old_samples = self.errors[:self.split_index].mean().item()
            ratio = mean_loss_old_samples / (loss_new_samples + 1e-6)
            percentage_new_samples = self.samples_in_buffer / self.dataset_size 
            if 1/max_oversampling_ratio - percentage_new_samples > 0 and loss_new_samples > mean_loss_old_samples:
                proposed_alpha = np.log( (1/max_oversampling_ratio - percentage_new_samples) / (1 - percentage_new_samples)) / np.log(ratio + 1e-6) 
                self.alpha = min(0.6, proposed_alpha)
            else:
                self.alpha = 0.6

                
    def compute_buffer_metrics(self):
        """Compute metrics for the prioritized replay buffer."""
        metrics = {
            'Buffer/priority_mean': self.priorities.mean().item() if self.alpha > 0.0 else 0.0,
            "Buffer/priority_std": self.priorities.std().item() if self.alpha > 0.0 else 0.0,
            "Buffer/priority_max": self.priorities.max().item() if self.alpha > 0.0 else 0.0,
            "Buffer/priority_min": self.priorities.min().item() if self.alpha > 0.0 else 0.0,
            "Buffer/beta": self.beta,
            'Buffer/alpha': self.alpha,
            "Buffer/average_weight": self.weights.mean().item() if self.alpha > 0.0 else 1.0,
            "Buffer/new_samples_in_buffer": self.samples_in_buffer,
            "Buffer/total_new_samples_seen": self.new_samples_count
        }
        return metrics
    
    
def seed_worker_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    