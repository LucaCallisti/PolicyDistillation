import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random
import os
import gymnasium as gym


class Visual_input():
    def __init__(self, init_screens, num_frames = 4, skipped_frames = 1):
        if init_screens.dim() == 3:
            init_screens = init_screens.unsqueeze(0)
            if init_screens.shape[0] != 1:
                raise ValueError("Dimension mismatch: expected a single env screen")
        # dim input = (num_envs, skipped_frames +1, nn_inputs, H, W)
        self.num_envs = init_screens.shape[0]
        self.nn_inputs = num_frames
        self.skipped_frames = skipped_frames

        self.input = torch.zeros((self.num_envs, (self.skipped_frames +1) * self.nn_inputs, init_screens.shape[2], init_screens.shape[3]), dtype=torch.float32)
        index = torch.arange(self.nn_inputs) * (self.skipped_frames + 1)
        self.indices = torch.zeros(self.skipped_frames+1, self.nn_inputs, dtype=torch.long)
        for i in range(self.skipped_frames + 1):
            self.indices[i] = (index + i) % ((self.skipped_frames +1) * self.nn_inputs)
        
        self.current_index = 0
        self.current_index_skipped_frame = 0
        
        self.input[:, :, :, :] = init_screens.expand(-1, (self.skipped_frames +1) * self.nn_inputs, -1, -1)

    def get_input(self):
        return self.input[:, self.indices[self.current_index_skipped_frame]]
    
    def update_deque(self, frames, dones=None):
        self.current_index_skipped_frame = (self.current_index_skipped_frame + 1) % (self.skipped_frames + 1)
        self.current_index = (self.current_index + 1) % ((self.skipped_frames +1) * self.nn_inputs)

        # Update the current frame
        self.input[:, self.current_index] = frames.squeeze()
        # For terminated environments, reinitialize everything
        
        if dones is not None:
            if isinstance(dones, np.ndarray):
                dones = torch.from_numpy(dones).bool()
            index = torch.arange(self.num_envs, device=dones.device)[dones.bool()]
            if len(index) > 0:
                selected_frames = frames[dones.bool()]
                if selected_frames.dim() == 5:
                    selected_frames = selected_frames.squeeze(1)
                self.input[index] = selected_frames.expand(-1, (self.skipped_frames +1) * self.nn_inputs, -1, -1)

        # Permutation to get the correct frame order
        reverse_index = torch.flip(torch.arange((self.skipped_frames +1) * self.nn_inputs), dims=[0])
        zero_first = torch.roll(reverse_index, shifts = 1)
        to_select = torch.flip(torch.roll(zero_first, shifts = self.current_index)[self.indices[0]], dims = [0])    # the last flip is needed because when trained, models had the most recent image as the last one

        return self.input[:, to_select]

def get_screen(envs = None, screen = None, frame_cfg = None):
    if envs is None and screen is None:
        raise ValueError("At least one between envs and screen must be provided")

    if frame_cfg is not None:
        shape = frame_cfg.get("shape", None)
        crop_index = frame_cfg.get("crop_index", None)
        grayscale = frame_cfg.get("grayscale", False)
        normalize = frame_cfg.get("normalize", False)
        interpolation = InterpolationMode.BICUBIC
    if screen is None:
        screen = envs.render()
    if isinstance(screen, tuple):
        screen = np.stack(screen)

    if isinstance(screen, np.ndarray):
        try:
            screen = torch.from_numpy(screen)
        except:
            screen = torch.from_numpy(screen.copy())
    if screen.dim() == 3:
        screen = screen.unsqueeze(0)
    screen = screen.permute(0, 3, 1, 2).float()
    if normalize:
        screen = screen / 255.0

    Transformation = []
    if crop_index is not None:
        top, bottom, left, right = crop_index
        screen = screen[:, :, top:bottom, left:right]
    
    if shape == -1 or shape is None:
        pass
    else:
        Transformation.append(T.Resize(shape, interpolation=interpolation))
    if grayscale:
        Transformation.append(T.Grayscale())
    Transformation_fun = T.Compose(Transformation) if len(Transformation) > 0 else None

    if Transformation_fun is not None:
        screen = Transformation_fun(screen)
    return screen


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon=1e-4, shape=(), device='cpu'):
        super().__init__()
        # Register as buffer for automatic .to(device)
        self.register_buffer('mean', torch.zeros(shape, device=device))
        self.register_buffer('var', torch.ones(shape, device=device))
        self.register_buffer('count', torch.tensor(epsilon, device=device))


    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.size(0)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count
        new_count = total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)

        if (self.var < 0.1).any():
            print("Warning: low variance in RunningMeanStd")


def set_seeds(seed, torch_deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # Extra determinism
    try:
        gym.utils.seeding.np_random(seed)
    except:
        pass    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA determinism
    torch.backends.cudnn.deterministic = torch_deterministic
