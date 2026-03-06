import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from Models.Utils import RunningMeanStd
import os
import wandb
import copy
from Models.Utils import get_screen, Visual_input


class BaseModel(nn.Module):
    def __init__(self, actor_net, critic_net, cnn_backbone, NormalizationObs, shape_to_normalize, action_type, input_type):
        super().__init__()
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.cnn_backbone = cnn_backbone if cnn_backbone is not None else nn.Identity()
        self.action_type = action_type
        if NormalizationObs:
            self.obs_rms = RunningMeanStd(shape=shape_to_normalize)
        self.input_type = input_type
        self.deterministic = False
        self._last_dist = None
        self._last_logits = None
        self.input_manager = None
        # Dictionary to store all inputs needed for model reconstruction
        self.model_init_args = {}
        self.max_log_std = 0.0
        self.min_log_std = -4.0

    def get_backbone(self, x):
        if hasattr(self, 'obs_rms'):
            x = (x - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
        x = self.cnn_backbone(x)
        x = x.view(x.size(0), -1)
        return x

    def get_value(self, x):
        if self.critic_net is None:
            raise RuntimeError("Critic is not defined for this model.")
        return self.critic_net(self.get_backbone(x))

    def get_action_continuous(self, x, action=None):
        mu, log_std = self.actor_net(x)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp()
        dist = Normal(mu, std)
        low, high = self.act_low.to(x.device), self.act_high.to(x.device)
        if self.deterministic:
            if action is None:
                raw_action = mu
                tanh_action = torch.tanh(raw_action)
                action = low + 0.5 * (tanh_action + 1.0) * (high - low)
            else:
                raise ValueError("In deterministic mode, action must be None.")
        if action is None:
            raw_action = dist.rsample()
            tanh_action = torch.tanh(raw_action)
            action_rescaled = low + 0.5 * (tanh_action + 1.0) * (high - low)
        else:
            action_rescaled = action
            tanh_action = 2 * (action_rescaled - low) / (high - low) - 1
            raw_action = atanh_stable(tanh_action)
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(torch.clamp(1 - tanh_action.pow(2), min=1e-6)).sum(-1)
        entropy = log_std.mean(-1)     
        self._last_dist = dist
        self._last_mu = mu
        self._last_log_std = log_std
        return action_rescaled, log_prob, entropy

    def get_action_discrete(self, x, action=None):
        logits = self.actor_net(x)
        logits = logits.clamp(-20, 20)
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

    def get_action(self, x, action=None):
        x = self.get_backbone(x)
        if self.action_type == "Discrete":
            return self.get_action_discrete(x, action)
        elif self.action_type == "Continuous":
            return self.get_action_continuous(x, action)

    def forward(self, x):
        return self.get_action(x)[0]

    def get_action_and_value(self, x, action=None):
        action, log_prob, entropy = self.get_action(x, action)
        value = self.get_value(x)
        return action, log_prob, entropy, value

    def get_logits(self):
        if self.action_type == "Discrete":
            return self._last_logits
        elif self.action_type == "Continuous":
            return torch.cat((self._last_mu, self._last_log_std), dim=-1)

    def create_distribution_from_logits(self, logits):
        if self.action_type == "Discrete":
            dist = Categorical(logits=logits)
        elif self.action_type == "Continuous":
            mu, log_std = torch.chunk(logits, 2, dim=-1)
            dist = Normal(mu, log_std.exp())
        return dist

    def get_last_distribution(self):
        if self._last_dist is None:
            raise RuntimeError("Call get_action_and_value() or get_action() first.")
        return self._last_dist

    def update_obs_rms(self, obs_batch):
        if hasattr(self, 'obs_rms'):
            self.obs_rms.update(obs_batch)
        else:
            raise RuntimeError("RunningMeanStd is not defined for this model.")

    def _create_save_dict(self, info_dict=None):
        dict_to_save = {
            "model_state_dict": self.state_dict(),
            "action_type": self.action_type,
            "agent_type": self.input_type,
            "environment_name": getattr(self, "env_name", "UnknownEnv"),
            "model_init_args": self.model_init_args if hasattr(self, "model_init_args") else {},
        }
        if hasattr(self, 'nn_inputs'):
            dict_to_save["nn_inputs"] = self.nn_inputs
        if hasattr(self, 'skipped_frames'):
            dict_to_save["skipped_frames"] = self.skipped_frames
        if hasattr(self, 'obs_rms'):
            dict_to_save["obs_rms"] = {
                "mean": self.obs_rms.mean,
                "var": self.obs_rms.var,
                "count": self.obs_rms.count,
            }
        if info_dict is not None:
            dict_to_save.update(info_dict)
        return dict_to_save

    def save_model(self, path, title="", wandb_bool=False, info_dict=None):
        os.makedirs(path, exist_ok=True)
        dict_to_save = self._create_save_dict(info_dict)
        filename = f"agent_{title}.pth" if title else "agent.pth"
        save_path = os.path.join(path, filename)
        torch.save(dict_to_save, save_path)
        if wandb_bool:
            try:
                wandb.save(save_path)
            except Exception as e:
                print(f"[WARNING] Could not save to wandb: {e}")
    @classmethod
    def load_from_dict(cls, dict_to_save):
        if 'mode' not in dict_to_save["model_init_args"]:
            for m in ['ImpaalaSmall', 'ImpaalaMid', 'ImpaalaBig']:
                try:
                    dict_to_save["model_init_args"]['mode'] = m
                    model = cls(**dict_to_save["model_init_args"])
                    model.load_state_dict(dict_to_save["model_state_dict"])
                    break
                except:
                    continue
        else:
            model = cls(**dict_to_save["model_init_args"])
            model.load_state_dict(dict_to_save["model_state_dict"])

        if "model_init_args" in dict_to_save:
            model.model_init_args = dict_to_save["model_init_args"]
        if "obs_rms" in dict_to_save and hasattr(model, "obs_rms"):
            model.obs_rms.mean = dict_to_save["obs_rms"]["mean"]
            model.obs_rms.var = dict_to_save["obs_rms"]["var"]
            model.obs_rms.count = dict_to_save["obs_rms"]["count"]
        if "nn_inputs" in dict_to_save:
            model.nn_inputs = dict_to_save["nn_inputs"]
        if "skipped_frames" in dict_to_save:
            model.skipped_frames = dict_to_save["skipped_frames"]
        return model

    def get_screen(self, envs=None, screen=None):
        frame_cfg = getattr(self, "frame_cfg", None)
        return get_screen(envs=envs, screen=screen, frame_cfg=frame_cfg)
    def reset_memory(self, init_screen):
        self.input_manager = Visual_input(init_screen, self.nn_inputs, self.skipped_frames)
        return self.input_manager.get_input()
    def update_memory(self, frame, dones):
        if self.input_manager is None:
            return self.reset_memory(frame)
        return self.input_manager.update_deque(frame, dones)
    def get_input_manager(self):
        aux = copy.deepcopy(self.input_manager)
        self.input_manager = None
        return aux
    def reset_input_manager(self):
        self.input_manager = None
    def set_input_manager(self, input_manager):
        self.input_manager = input_manager


def atanh_stable(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if torch.any(x < -1) or torch.any(x > 1):
        raise ValueError("All elements must be in [-1, 1]")
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * torch.log1p(2 * x / (1 - x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())