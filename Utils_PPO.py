from operator import index
import random
import numpy as np
import torch
import gymnasium as gym
from My_env import my_LunarLander
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import os
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import wandb
import cv2
from typing import Optional


def make_env(idx, capture_video, run_name, folder_path, enviroment, all_render = False):
    def thunk():
        if capture_video and idx == 0:
            if enviroment == "LunarLander":
                env = my_LunarLander(render_mode='rgb_array')
            elif enviroment == 'CarRacing':
                env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous = False)
            else:
                raise NotImplementedError(f"Environment {enviroment} not implemented.")
            env = gym.wrappers.RecordVideo(env,  os.path.join(folder_path, "videos", run_name), episode_trigger=lambda episode_id: episode_id % 5 == 3)
        elif all_render:
            if enviroment == "LunarLander":
                env = my_LunarLander(render_mode='rgb_array')
            elif enviroment == 'CarRacing':
                env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous = False)
            else:
                raise NotImplementedError(f"Environment {enviroment} not implemented.")
        else:
            if enviroment == "LunarLander":
                env = my_LunarLander()
            elif enviroment == 'CarRacing':
                env = gym.make('CarRacing-v3', continuous = False)
            else:
                raise NotImplementedError(f"Environment {enviroment} not implemented.")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO_Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        if hasattr(envs, "single_observation_space"):
            obs_space = envs.single_observation_space
            act_space = envs.single_action_space
        else:
            obs_space = envs.observation_space
            act_space = envs.action_space

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_space.n), std=0.01),
        )
        self.agent_type = 'State'

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action = None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            if self.training:
                action = probs.sample()
            else:
                action = logits.argmax(dim=-1)
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def forward(self, x):
        self.logits = self.actor(x) 
        self.last_probs = torch.softmax(self.logits, dim=-1)
        if self.training:
            action = torch.multinomial(self.last_probs, num_samples=1).squeeze(-1)
        else:
            action = self.logits.argmax(dim=-1)
        return action, self.logits

    def get_logprobs(self):
        # return self.last_probs.logits   # i logits sono i logprob
        return torch.log(self.last_probs + 1e-8)   # Aggiungi una piccola costante per evitare log(0)

    def get_entropy(self):
        # return self.last_probs.entropy()
        return -(self.last_probs * torch.log(self.last_probs + 1e-8)).sum(dim=-1)  # Entropia calcolata manualmente

    def get_logits(self):
        return self.logits

    def device(self):
        return next(self.parameters()).device  




def get_cnn_output_size(cnn, h, w, in_channels=1, device='cpu'):
    x = torch.zeros(1, in_channels, h, w).to(device)
    with torch.no_grad():
        x = cnn(x)
    return x.view(-1).shape[0]
class Image_model(nn.Module):
    def __init__(self, env, num_frames = 4, skipped_frames = 1):
        super().__init__()
        self.nn_inputs = num_frames
        self.skipped_frames = skipped_frames
        HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3 = 32, 64, 128
        KERNEL_SIZE = 5
        STRIDE = 1  

        self.CNN = nn.Sequential(
            nn.Conv2d(self.nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(HIDDEN_LAYER_3, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_3),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_LAYER_3, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        frame = get_screen(env)
        h, w = frame.shape[2], frame.shape[3]
        linear_input_size = get_cnn_output_size(self.CNN, h, w, in_channels=num_frames, device='cpu')
        print("linear_input_size", linear_input_size)
        self.fc2 = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        self.output_size = env.action_space.n
        self.agent_type = 'Image'

    def forward(self, x):
        out = self.CNN(x) 
        out = out.view(out.size(0), -1)  
        self.logits = self.fc2(out)  

        self.last_probs = torch.softmax(self.logits, dim=-1)
        if self.fc2.training:
            action = torch.multinomial(self.last_probs, num_samples=1).squeeze(-1)
        else:
            action = self.logits.argmax(dim=-1)
        return action, self.logits
        

    def get_logits(self):
        return self.logits
    
    def get_logprobs(self):
        # return self.last_probs.logits   # i logits sono i logprob
        return torch.log(self.last_probs + 1e-8)  # Aggiungi una piccola costante per evitare log(0)
    
    def get_entropy(self):
        # return self.last_probs.entropy()
        return -(self.last_probs * torch.log(self.last_probs + 1e-8)).sum(dim=-1)  # Entropia calcolata manualmente
 
    def get_screen(self, env):
        return get_screen(env)
    def init_deque(self, init_screen):
        self.input = Visual_input(init_screen, self.nn_inputs, self.skipped_frames)
        return self.input.get_input()
    def append_deque(self, frame, dones=None):
        if dones is None:
            dones = torch.zeros(frame.shape[0], dtype=torch.bool)
        return self.input.update_deque(frame, dones)
    
    def reinitialize_fc2(self):
        """Reinizializza il layer fc2"""
        for layer in self.fc2:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.0)
                print(f"Reinizializzato fc2 layer: {layer}")
    
    def save_model(self, mean_ID_reward, seed_Id, mean_OD_reward, seed_OD, epoch, path, state_model_dict = None, title='', wandb_bool=False):
        dict_to_save = {
            'num_frames': self.nn_inputs,
            'skipped_frames': self.skipped_frames,
            'mean_ID_reward': mean_ID_reward,
            'mean_OD_reward': mean_OD_reward,
            'seed_Id': torch.tensor(seed_Id),
            'seed_OD': torch.tensor(seed_OD),
            'epoch': epoch
        }
        if state_model_dict is None:
            dict_to_save['model_state_dict'] = self.state_dict()
        else:
            dict_to_save['model_state_dict'] = state_model_dict
        path = os.path.join(path, f"student_model_distilled_{title}.pth")
        torch.save(dict_to_save, path)
        if wandb_bool:
            wandb.save(path)

    @staticmethod
    def load_model_from_path(path, env, device='cpu', info = False):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if 'num_frames' not in checkpoint:
            model = Image_model(env, num_frames=4, skipped_frames=1).to(device)
            model.load_state_dict(checkpoint, strict=True)
        else:
            num_frames = checkpoint['num_frames']
            skipped_frames = checkpoint['skipped_frames']
            model = Image_model(env, num_frames=num_frames, skipped_frames=skipped_frames).to(device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if info:
                print("Loaded model:")
                for key, value in checkpoint.items():
                    if key not in ['model_state_dict']:
                        print(f"  {key}: {value}")
        return model
    

class PPO_Image_model(nn.Module):
    def __init__(self, envs, num_frames = 4, skipped_frames = 1):
        super().__init__()
        self.nn_inputs = num_frames
        self.skipped_frames = skipped_frames
        HIDDEN_LAYER_1, HIDDEN_LAYER_2, HIDDEN_LAYER_3 = 32, 64, 128
        KERNEL_SIZE = 5
        STRIDE = 1  

        if hasattr(envs, "single_observation_space"):
            act_space = envs.single_action_space
        else:
            act_space = envs.action_space
        
        try:
            self.num_envs = envs.num_envs
        except:
            self.num_envs = 1
        envs.reset()

        self.CNN = nn.Sequential(
            nn.Conv2d(self.nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_1),
            nn.ReLU(),           
            nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_2),
            nn.ReLU(),          
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_2),
            nn.ReLU(),          
            nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_3),
            nn.ReLU(),          
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(HIDDEN_LAYER_3, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_3),
            nn.ReLU(),          
            nn.Conv2d(HIDDEN_LAYER_3, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm2d(HIDDEN_LAYER_3),
            nn.ReLU(),         
            nn.AdaptiveAvgPool2d(1)
        )
  
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_3, 128),
            nn.ReLU(),
            nn.Linear(128, act_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.output_size = act_space.n
        self.agent_type = 'Image'

    def forward(self, x):
        out = self.CNN(x) 
        out = out.view(out.size(0), -1)  
        self.logits = self.fc2(out)  

        self.last_probs = torch.softmax(self.logits, dim=-1)
        if self.training:
            action = torch.multinomial(self.last_probs, num_samples=1).squeeze(-1)
        else:
            action = self.logits.argmax(dim=-1)
        return action, self.logits
    
    def get_value(self, x):
        out = self.CNN(x) 
        out = out.view(out.size(0), -1)  
        value = self.critic(out)
        return value
    def get_action_and_value(self, x, action=None):
        out = self.CNN(x) 
        out = out.view(out.size(0), -1)  
        self.logits = self.actor(out)  

        self.last_probs = torch.softmax(self.logits, dim=-1)
        if action is None:
            if self.actor.training:
                action = torch.multinomial(self.last_probs, num_samples=1).squeeze(-1)
            else:
                action = self.logits.argmax(dim=-1)
   

        logprob = self.get_logprobs()[torch.arange(action.shape[0]), action]
        return action, logprob, self.get_entropy(), self.critic(out)

    def get_logits(self):
        return self.logits

    def get_logprobs(self):
        return torch.log(self.last_probs + 1e-8)  # Aggiungi una piccola costante per evitare log(0)
    def get_entropy(self):
        return -(self.last_probs * torch.log(self.last_probs + 1e-8)).sum(dim=-1)  # Entropia calcolata manualmente
    
    def get_screen(self, env):
        return get_screen(env)
    def init_deque(self, init_screen):
        self.input = Visual_input(init_screen, self.nn_inputs, self.skipped_frames)
        return self.input.get_input()
    def append_deque(self, frame, dones):
        return self.input.update_deque(frame, dones)

    def save_model(self, mean_ID_reward, seed_Id, mean_OD_reward, seed_OD, epoch, path, title='', wandb_bool=False):
        dict_to_save = {
            'model_state_dict': self.state_dict(),
            'num_frames': self.nn_inputs,
            'skipped_frames': self.skipped_frames,
            'num_envs': self.num_envs,
            'mean_ID_reward': mean_ID_reward,
            'mean_OD_reward': mean_OD_reward,
            'seed_Id': torch.tensor(seed_Id),
            'seed_OD': torch.tensor(seed_OD),
            'epoch': epoch
        }
        path = os.path.join(path, f"student_model_distilled_{title}.pth")
        torch.save(dict_to_save, path)
        if wandb_bool:
            wandb.save(path)

    @staticmethod
    def load_model_from_path(path, env, device='cpu', info = False):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if 'num_frames' not in checkpoint:
            model = Image_model(env, num_frames=4, skipped_frames=1).to(device)
            model.load_state_dict(checkpoint)
        else:
            num_frames = checkpoint['num_frames']
            skipped_frames = checkpoint['skipped_frames']
            model = Image_model(env, num_frames=num_frames, skipped_frames=skipped_frames).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if info:
                print("Loaded model:")
                for key, value in checkpoint.items():
                    if key not in ['model_state_dict']:
                        print(f"  {key}: {value}")
        return model

        
class Visual_input():
    def __init__(self, init_screens, num_frames = 4, skipped_frames = 1):
        if init_screens.dim() == 3:
            init_screens = init_screens.unsqueeze(0)
            assert self.num_envs == 1, "Dimension mismatch"
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

        # Aggiorna il frame corrente
        self.input[:, self.current_index] = frames.squeeze()
        # Per ambienti terminati, reinizializza tutto
        
        if dones is not None:
            if isinstance(dones, np.ndarray):
                dones = torch.from_numpy(dones).bool()
        index = torch.arange(self.num_envs, device=dones.device)[dones.bool()]
        if len(index) > 0:
            selected_frames = frames[dones.bool()]
            if selected_frames.dim() == 5:
                selected_frames = selected_frames.squeeze(1)
            self.input[index] = selected_frames.expand(-1, (self.skipped_frames +1) * self.nn_inputs, -1, -1)

        # permutazione per avere l'ordine corretto dei frame
        reverse_index = torch.flip(torch.arange((self.skipped_frames +1) * self.nn_inputs), dims=[0])
        zero_first = torch.roll(reverse_index, shifts = 1)
        to_select = torch.flip(torch.roll(zero_first, shifts = self.current_index)[self.indices[0]], dims = [0])    # l'ultimo flip serve perchè quando ho allenato i modelli avevano come ultima l'immagine più recente

        return self.input[:, to_select]

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


class CreateBuffer_fromEpisode():
    def __init__(self, my_dict):
        self.frames = my_dict['frames']
        self.states = my_dict['states']
        self.logits_actions = my_dict['logits_actions']
        self.seed_run = my_dict['seed_run']
        self.new_episode_index = my_dict['new_episode_index']
        self.current_index = len(self.frames)

    def CreateBuffer(self, num_frames, skipped_frames, wandb_bool=True, create_od_set = False):
        # se num_frames = 3, skipped_frames = 2 allora gli indici saranno
        # [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 4], [0, 2, 5], [0, 3, 6], [1, 4, 7]]
        
        # Calcola le dimensioni degli episodi
        episode_ends = torch.cat([self.new_episode_index[1:], torch.tensor([self.current_index])])
        episode_starts = self.new_episode_index
        episode_lengths = episode_ends - episode_starts
        max_ep_len = episode_lengths.max().item()
        
        # Calcola il numero totale di sequenze che genereremo (approssimativo)
        # Useremo una sovrastima e poi troncaremo alla fine
        total_sequences_estimate = episode_lengths.sum().item()
        
        # Pre-alloca i tensori con le dimensioni stimate (potrebbe essere sovrastimato)
        frame_shape = self.frames.shape[1:]  # (H, W) o shape senza batch
        state_shape = self.states.shape[1:]  # (state_dim,)
        action_shape = self.logits_actions.shape[1:]  # (action_dim,)
        
        buffer_tensor = torch.zeros(total_sequences_estimate, num_frames, *frame_shape, dtype=self.frames.dtype)
        states_tensor = torch.zeros(total_sequences_estimate, *state_shape, dtype=self.states.dtype)
        actions_tensor = torch.zeros(total_sequences_estimate, *action_shape, dtype=self.logits_actions.dtype)
        
        # Pre-calcola tutti gli indici per l'episodio più lungo
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

        current_seq_idx = 0  # Indice della sequenza corrente
        
        if create_od_set:
            self.OD_set = {}

        # Processa ogni episodio riempiendo direttamente i tensori pre-allocati
        for ep_start, ep_end in zip(episode_starts, episode_ends):
            ep_frames = self.frames[ep_start:ep_end]
            ep_states = self.states[ep_start:ep_end]
            ep_actions = self.logits_actions[ep_start:ep_end]
            ep_len = len(ep_frames)
            
            # Per ogni frame dell'episodio - VERSIONE VETTORIZZATA
            indices_tensor = torch.tensor(all_frame_indices[:ep_len])  # Prendi solo gli indici validi per questo episodio

            selected_frames = ep_frames[indices_tensor]  # Shape: [num_valid, num_frames, H, W]
            selected_states = ep_states[indices_tensor[:, -1]]  # Ultimo frame di ogni sequenza
            selected_actions = ep_actions[indices_tensor[:, -1]]  # Ultimo frame di ogni sequenza
            
            # Riempi i tensori pre-allocati
            buffer_tensor[current_seq_idx:current_seq_idx + ep_len] = selected_frames
            states_tensor[current_seq_idx:current_seq_idx + ep_len] = selected_states
            actions_tensor[current_seq_idx:current_seq_idx + ep_len] = selected_actions
            
            current_seq_idx += ep_len

            if create_od_set and len(self.OD_set) < 10:
                seed = self.seed_run[ep_start].item()
                if seed not in self.OD_set:
                    self.OD_set[seed] = {
                        'frames': ep_frames,
                        'states': ep_states,
                        'logits_actions': ep_actions,
                        'rewards': self.rewards[ep_start]
                    }

        print(f"Sequenze generate: {current_seq_idx}/{total_sequences_estimate}")

        # Non dovrebbe essere necessario
        self.buffer_tensor = buffer_tensor[:current_seq_idx]
        self.states_tensor = states_tensor[:current_seq_idx]
        self.actions_tensor = actions_tensor[:current_seq_idx] 

        if wandb_bool:
            imgs = [self.buffer_tensor[555, i] for i in range(self.buffer_tensor.shape[1])]
            imgs_formatted = []
            for i, img in enumerate(imgs):
                if img.dim() == 2:  # Se è 2D (H, W), aggiungi dimensione canali
                    img = img.unsqueeze(0)  # Diventa (1, H, W)
                imgs_formatted.append(wandb.Image(img, caption=f"frame_{i}"))
            wandb.log({"buffer_images": imgs_formatted})  

    def save_OD_set(self, path, title='', wandb_bool=False):
        if not hasattr(self, 'OD_set'):
            raise ValueError("OD_set non creato. Esegui prima CreateBuffer con create_od_set=True.")
        
        path = os.path.join(path, f"OD_set_{title}.pth")
        torch.save(self.OD_set, path)
        if wandb_bool:
            wandb.save(path)
        print(f"OD_set salvato in {path}")

    def Dataloader(self, batch_size=32, gaussian_noise_std=0.0):
        """Crea un DataLoader per i tensori pre-allocati, con opzionale augmentation Gaussian noise (solo torchvision)"""
        if not hasattr(self, 'buffer_tensor'):
            raise ValueError("Buffer non creato. Esegui prima CreateBuffer.")

        if gaussian_noise_std > 0.0:
            from torchvision.transforms.v2 import GaussianNoise
            noise_transform = GaussianNoise(sigma=gaussian_noise_std)
            def collate_with_noise(batch):
                buffers, states, actions = zip(*batch)
                buffers = torch.stack(buffers)
                states = torch.stack(states)
                actions = torch.stack(actions)
                buffers = noise_transform(buffers)
                return buffers, states, actions
            collate_fn = collate_with_noise
            imgs = [self.buffer_tensor[555, i] for i in range(self.buffer_tensor.shape[1])]
            imgs_formatted = []
            for i, img in enumerate(imgs):
                if img.dim() == 2:  # Se è 2D (H, W), aggiungi dimensione canali
                    img = img.unsqueeze(0)  # Diventa (1, H, W)
                imgs_formatted.append(wandb.Image(noise_transform(img), caption=f"frame_{i}"))
            wandb.log({"buffer_images_with_noise": imgs_formatted})        
        else:
            collate_fn = None
        
        dataset = torch.utils.data.TensorDataset(self.buffer_tensor, self.states_tensor, self.actions_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=collate_fn
        )
        return dataloader


class TensorAugmentation(nn.Module):
    def __init__(self, augment_prob=0.4):
        super().__init__()
        self.augment_prob = augment_prob
        self.tensor_augment_transforms = nn.Sequential(
                # Rotazione su tensor
                T.RandomRotation(degrees=1.5),
                
                # Traslazione su tensor  
                T.RandomAffine(degrees=0, translate=(0.02, 0.015)),
            )        
    def forward(self, x):
        # x ha shape [batch_size, channels, height, width]
        if torch.rand(1) < self.augment_prob:
            return self.tensor_augment_transforms(x)

def get_screen(envs = None, screen = None):
    if envs is None and screen is None:
        raise ValueError("At least one between envs and screen must be provided")
    if screen is None:
        screen = envs.render()
    if isinstance(screen, tuple):
        screen = np.stack(screen)
    if isinstance(screen, np.ndarray):
        screen = torch.from_numpy(screen)
    if screen.dim() == 3:
        screen = screen.unsqueeze(0)
    screen = screen.permute(0, 3, 1, 2).float() / 255.0

    shape = (min(screen.shape[2], 133), min(screen.shape[3], 200))
    resize_fun = T.Compose([
        T.Resize(shape, interpolation=InterpolationMode.BICUBIC),  # original size is (400, 600)
        T.Grayscale(),
        ])
    
    screen = resize_fun(screen)
    return screen

    

# Da testare
class EpisodeVideoRecorder:
    """Simple class for creating videos from episode frames"""
    
    def __init__(self, output_dir: str = "./videos", fps: int = 50, codec: str = "mp4v"):
        """
        Initialize the video recorder
        
        Args:
            output_dir: Directory where videos will be saved
            fps: Frames per second in the output video
            codec: Video codec (mp4v, XVID, etc.)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.frames = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def add_frame(self, frame: np.ndarray):
        """
        Add a single frame to the buffer
        
        Args:
            frame: Frame as numpy array (H, W, 3) with values [0-255] or [0-1]
        """
        # Normalize if necessary
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        self.frames.append(frame)
    
    def save_video(self, episode_name: str, run_name: str = "episode", wandb_bool: bool = True) -> Optional[str]:
        """
        Save accumulated frames as a video file
        
        Args:
            episode_name: Name of the episode (e.g., "ep_001_reward_250")
            run_name: Name of the run
            wandb_bool: If True, save the video to wandb as well
            
        Returns:
            Path to the created video, or None if an error occurred
        """
        if len(self.frames) == 0:
            print("Warning: No frames available")
            return None
        
        # Get dimensions from the first frame
        height, width = self.frames[0].shape[:2]
        
        # Create video file path
        video_filename = f"{run_name}_{episode_name}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Unable to create video {video_path}")
            return None
        
        # Write all frames to the video file
        for frame in self.frames:
            # Convert from RGB to BGR if necessary (OpenCV uses BGR)
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        out.release()
        self.frames = []  # Reset after saving
        print(f"Video saved: {video_path}")
        
        # Save to wandb if enabled
        if wandb_bool:
            wandb.log({
                "episode_video": wandb.Video(video_path, format="mp4")
            })
        
        return video_path