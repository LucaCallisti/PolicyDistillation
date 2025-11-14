import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical, Normal
import wandb
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from Utils_PPO import set_seeds
import copy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, actor_layers=[128, 128], critic_layers=None, cnn_channels = None, seed = 0):
        super().__init__()
        set_seeds(seed)

        # --- gestione compatibilità VecEnv / Env singolo ---
        if hasattr(envs, "single_observation_space"):
            obs_space = envs.single_observation_space
            act_space = envs.single_action_space
        else:
            obs_space = envs.observation_space
            act_space = envs.action_space

        obs_dim = np.array(obs_space.shape).prod()
        self.env_name = getattr(envs, "spec", None).id if hasattr(envs, "spec") and envs.spec is not None else str(envs)

        # --- riconosci tipo di action space ---
        if isinstance(act_space, gym.spaces.Discrete):
            self.action_type = "Discrete"
            act_dim = act_space.n
        elif isinstance(act_space, gym.spaces.Box):
            self.action_type = "Continuous"
            act_dim = np.prod(act_space.shape)
            self.act_low = torch.tensor(act_space.low, dtype=torch.float32)
            self.act_high = torch.tensor(act_space.high, dtype=torch.float32)
        else:
            raise NotImplementedError("Tipo di action space non supportato.")
        
        # -- CNN backbone (se necessario) ---
        if cnn_channels is not None:
            self.nn_inputs = 4
            self.skipped_frames = 1
            layers = []
            in_channels = self.nn_inputs
            for layer in cnn_channels:
                if layer[0] == "conv":
                    _, out_channels, kernel_size, stride = layer
                    layers.append(layer_init(nn.Conv2d(in_channels, out_channels, kernel_size, stride)))
                    in_channels = out_channels
                elif layer[0] == "BatchNorm":
                    layers.append(nn.BatchNorm2d(in_channels))
                elif layer[0] == "GroupNorm":
                    layers.append(nn.GroupNorm(in_channels // 4, in_channels))
                elif layer[0] == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif layer[0] == "pool":
                    _, pool_type, *params = layer
                    if pool_type == "max":
                        layers.append(nn.MaxPool2d(*params))
                    elif pool_type == "adaptive":
                        layers.append(nn.AdaptiveAvgPool2d(*params))
                    else:
                        raise ValueError(f"Unknown pool type: {pool_type}")

            self.cnn_backbone = nn.Sequential(*layers)

            # calcolo automatico dimensione output CNN
            with torch.no_grad():
                screen, _ = envs.reset()
                dummy_input = Visual_input(get_screen(screen = screen)).get_input()
                self.input_shape = dummy_input.squeeze().shape
                cnn_out = self.cnn_backbone(dummy_input)
                self.flat_dim = cnn_out.view(1, -1).shape[1]
            self.input_manager = None
            

        # --- costruzione dinamica Critic ---
        self.critic_mode = (critic_layers is not None)
        if self.critic_mode:
            critic_layers_list = []
            last_dim = obs_dim if cnn_channels is None else self.flat_dim
            for h in critic_layers:
                critic_layers_list += [layer_init(nn.Linear(last_dim, h)), nn.Tanh()]
                last_dim = h
            critic_layers_list.append(layer_init(nn.Linear(last_dim, 1), std=1.0))
            self.critic = nn.Sequential(*critic_layers_list)
        
        # --- costruzione dinamica Actor backbone ---
        actor_layers_list = []
        last_dim = obs_dim if cnn_channels is None else self.flat_dim
        for h in actor_layers:
            actor_layers_list += [layer_init(nn.Linear(last_dim, h)), nn.Tanh()]
            last_dim = h
        self.actor_backbone = nn.Sequential(*actor_layers_list)
        self.input_shape = (int(obs_dim),) if cnn_channels is None else self.input_shape

        # --- head finale dell’attore ---
        if self.action_type == "Discrete":
            self.actor_out = layer_init(nn.Linear(last_dim, act_dim), std=0.01)
        else:
            self.mu_layer = layer_init(nn.Linear(last_dim, act_dim), std=0.01)
            self.log_std_param = layer_init(nn.Linear(last_dim, act_dim))
            torch.nn.init.constant_(self.log_std_param.weight, 0.0)
            torch.nn.init.constant_(self.log_std_param.bias, np.log(0.5))  

        # --- info ---
        self.agent_type = "State" if cnn_channels is None else "Image"
        self.deterministic = False
        self._last_dist = None
        self._last_logits = None
        self._architecture_info = {
            "actor_layers": actor_layers,
            "critic_layers": critic_layers,
            "cnn_channels": cnn_channels,
        }

    def get_backbone(self, x):
        if self.agent_type == "Image":
            x = self.cnn_backbone(x)
            x = x.view(x.size(0), -1)
            return x
        return x

    def get_value(self, x):
        """Restituisce il valore stimato V(s)."""
        if not self.critic_mode:
            raise RuntimeError("Critic non definito per questo modello.")
        return self.critic(self.get_backbone(x))

    def get_action(self, x, action=None):
        """Restituisce (azione, log_prob, entropia)."""
        temp = x
        x = self.get_backbone(x)
        if self.action_type == "Discrete":
            logits = self.actor_out(self.actor_backbone(x))
            dist = Categorical(logits=logits)
            if self.deterministic:
                if action is None:
                    action = torch.argmax(logits, dim=-1)
                else:
                    raise ValueError("In deterministic mode, action must be None.")
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            self._last_dist = dist
            self._last_logits = logits
            return action, log_prob, entropy

        # Continuous
        else:
            x = self.actor_backbone(x) 
            mu = self.mu_layer(x) 
            log_std = self.log_std_param(x).clamp(-20, 2)
            std = log_std.exp() 
            dist = Normal(mu, std)  
            if self.deterministic: 
                if action is None: 
                    self._last_dist = dist 
                    action = mu
                else: 
                    raise ValueError("In deterministic mode, action must be None.") 
            low, high = self.act_low.to(x.device), self.act_high.to(x.device) 
            if action is None: 
                raw_action = dist.rsample() 
                tanh_action = torch.tanh(raw_action) 
                action_rescaled = low + 0.5 * (tanh_action + 1.0) * (high - low) 
            else: 
                action_rescaled = action 
                tanh_action = 2 * (action_rescaled - low) / (high - low) - 1 
                raw_action = torch.atanh(tanh_action.clamp(-0.999, 0.999)) 
                
            # log_prob corretto 
            log_prob = dist.log_prob(raw_action).sum(-1) 
            log_prob -= torch.log(torch.clamp(1 - tanh_action.pow(2), min=1e-6)).sum(-1)
            entropy = dist.entropy().sum(-1)
            self._last_dist = dist 
            self._last_mu = mu 
            self._last_std = std 
            return action_rescaled, log_prob, entropy
    
    def forward(self, x):
        return self.get_action(x)[0]

    def get_action_and_value(self, x, action=None):
        """Restituisce (azione, log_prob, entropia, valore)."""
        action, log_prob, entropy = self.get_action(x, action)
        value = self.get_value(x)
        return action, log_prob, entropy, value

    def get_logits(self):
        """Restituisce ciò che è necessario per ricreare le distribuzioni."""
        if self.action_type == "Discrete":
            return self._last_logits
        elif self.action_type == "Continuous":
            return self._last_mu, self._last_std
        
    def create_distribution_from_logits(self, logits):
        """Crea una distribuzione a partire dai logit."""
        if self.action_type == "Discrete":
            dist = Categorical(logits=logits)
        elif self.action_type == "Continuous":
            mu, std = logits
            dist = Normal(mu, std)
        return dist

    def get_last_distribution(self):
        """Restituisce i log-prob dell’ultima distribuzione."""
        if self._last_dist is None:
            raise RuntimeError("Chiamare prima get_action_and_value() o get_action().")
        return self._last_dist


    def get_entropy(self):
        """Restituisce l'entropia dell'ultima distribuzione."""
        if self._last_dist is None:
            raise RuntimeError("Chiamare prima get_action_and_value() o get_action().")
        if self.action_type == "Discrete":
            return self._last_dist.entropy()
        return self._last_dist.entropy().sum(-1)

    def device(self):
        return next(self.parameters()).device
    
    def get_screen(self, envs=None, screen=None):
        return get_screen(envs, screen)
    def reset_memory(self, init_screen):
        self.input_manager = Visual_input(init_screen, self.nn_inputs, self.skipped_frames)
        return self.input_manager.get_input()
    def update_memory(self, frame, dones):
        if self.input_manager is None:
            return self.reset_memory(frame)
        return self.input_manager.update_deque(frame, dones)
    def get_input_manager(self):
        return copy.deepcopy(self.input_manager)
    def set_input_manager(self, input_manager):
        self.input_manager = input_manager
    
    def save_model(self, path, title="", wandb_bool=False, info_dict=None):
        """
        Salva il modello su file, includendo lo state_dict e informazioni aggiuntive.

        Args:
            path (str): percorso della directory in cui salvare il modello.
            title (str): nome aggiuntivo per il file .pth.
            wandb_bool (bool): se True, salva anche il file su Weights & Biases.
            info_dict (dict): dizionario contenente info extra (es. rewards, seed, epoch...).
        """
        os.makedirs(path, exist_ok=True)

        dict_to_save = {
            "model_state_dict": self.state_dict(),
            "action_type": self.action_type,
            "agent_type": self.agent_type,
            "critic_mode": self.critic_mode,
            "environment_name": getattr(self, "env_name", "UnknownEnv"),  # ← nome dell’ambiente
            "architecture_info": self._architecture_info,
        }
        if self.agent_type == "Image":
            dict_to_save["nn_inputs"] = self.nn_inputs
            dict_to_save["skipped_frames"] = self.skipped_frames

        if info_dict is not None:
            dict_to_save.update(info_dict)

        filename = f"ppo_agent_{title}.pth" if title else "ppo_agent.pth"
        save_path = os.path.join(path, filename)
        torch.save(dict_to_save, save_path)

        if wandb_bool:
            try:
                wandb.save(save_path)
            except Exception as e:
                print(f"[WARNING] Impossibile salvare su wandb: {e}")

    @staticmethod
    def load_model_from_path(path, env, device="cpu", info=False):
        """
        Carica un modello PPO_Agent da file .pth.

        Args:
            path (str): percorso del file da caricare.
            env (gym.Env): ambiente associato al modello.
            device (str): 'cpu' o 'cuda'.
            info (bool): se True, stampa le informazioni salvate.
        """
        checkpoint = torch.load(path, map_location=device)
        architecture_info = checkpoint.get("architecture_info", {})

        model = Agent(env, **architecture_info).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if info:
            for key, value in checkpoint.items():
                if key != "model_state_dict":
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