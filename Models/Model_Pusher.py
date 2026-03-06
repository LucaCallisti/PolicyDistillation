import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from Enviroment.My_wrapper import AddFrameObsWrapper, AddPartialStateObsWrapper, AddStateObsWrapper, Float32ObsWrapper
from Models.Utils import get_screen, Visual_input, set_seeds
from Models.Base_Model import BaseModel, count_parameters
from Enviroment.Utils import make_env
from Distillation.Utils import get_dict_envs
from Models.Impoola import Imapala_Backbone, build_continuous_actor, build_critic, layer_init_orthogonal
import numpy as np

def _get_wrappers(mode):
    if mode == 'State':
        wrappers = [Float32ObsWrapper, AddStateObsWrapper]
    elif 'Impaala' in mode:
        wrappers = [Float32ObsWrapper, AddStateObsWrapper, AddFrameObsWrapper, AddPartialStateObsWrapper]
    return wrappers

def get_Teacher_model(Teacher_path = None, Vecnorm_path=None, device = 'cpu'):
    if Teacher_path is None:
        Teacher_path = './Results/Pusher/Teacher/sac_checkpoint_1660000_steps' 
    if Vecnorm_path is None:
        Vecnorm_path = './Results/Pusher/Teacher/sac_checkpoint_vecnormalize_1660000_steps.pkl' 
    Teacher = Model_from_SB(model_path =Teacher_path, vecnorm_path=Vecnorm_path, env = gym.make("Pusher-v5"), device=device)
    return Teacher

class StudentModelPusher(BaseModel):
    def __init__(self, mode='Impaala', critic_net=False, NonLinearity=nn.SiLU, NormalizationObs=False, seed=0):
        set_seeds(seed)
        path_folder = ''
        dict_enviroment, dict_test_enviroment = get_dict_envs(mode, path_folder, wrappers=_get_wrappers(mode), run_name='', env_name='Pusher-v5')
        env = make_env(dict_test_enviroment, 0, wrappers=dict_test_enviroment['wrappers'])()
        obs, info = env.reset()
        if 'Impaala' in mode:
            self.nn_inputs = 4
            self.skipped_frames = 1
            # self.frame_cfg = {
            #     "crop_index": (230, 380, 50, -50),
            #     "shape": -1,
            #     "grayscale": True,
            #     "normalize": True,
            # }
            self.frame_cfg = {
                "crop_index": (70, 230, 70, 410),
                "shape": (64, 128),
                "grayscale": True,
                "normalize": True,
            }
            dummy_input = Visual_input(self.get_screen(screen=info['Frame']), num_frames = self.nn_inputs).get_input()
            shape = dummy_input.squeeze(0).shape
            self.check_shape = shape
            Adaptive_pooling_size = 3
            if mode == 'ImpaalaMid' or mode == 'Impaala':
                last_channels = 64
                cnn_backbone = Imapala_Backbone(shape=shape, cnn_filters=(32, 64, last_channels), activation=nn.ReLU, use_AvgPool=True, pooling_size=Adaptive_pooling_size)          
            elif mode == 'ImpaalaSmall':
                last_channels = 32
                cnn_backbone = Imapala_Backbone(shape=shape, cnn_filters=(16, 32, last_channels), activation=nn.ReLU, use_AvgPool=True, pooling_size=Adaptive_pooling_size)
            elif mode == 'ImpaalaBig':
                last_channels = 128
                cnn_backbone = Imapala_Backbone(shape=shape, cnn_filters=(64, 128, last_channels), activation=nn.ReLU, use_AvgPool=True, pooling_size=Adaptive_pooling_size)
            # actor_in_dim = last_channels + info['PartialState'].shape[0]
            actor_in_dim = last_channels * Adaptive_pooling_size**2 + 64
            self.input_shape = {'Frame': shape, 'PartialState' : info['PartialState'].shape}
            self.input_type = ["Frame", 'PartialState']
        else:
            cnn_backbone = None
            actor_in_dim = int(np.prod(env.observation_space.shape))
            self.input_shape = {'State': (actor_in_dim,)}
            self.input_type = ["State"]

        actor_net = build_continuous_actor(actor_in_dim, env.action_space.shape[0], NonLinearity=NonLinearity)
        critic_net = build_critic(actor_in_dim, NonLinearity=NonLinearity) if critic_net else None
        env.close()
        super().__init__(actor_net, critic_net, cnn_backbone=cnn_backbone, NormalizationObs=NormalizationObs, shape_to_normalize=env.observation_space.shape, action_type="Discrete", input_type=self.input_type)
        self.input_manager = None
        self.action_type = "Continuous"
        self.act_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.act_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.model_init_args = {
            'mode': mode,
            'critic_net': critic_net,
            'NonLinearity': NonLinearity,
            'NormalizationObs': NormalizationObs,
            'seed': seed
        }

        self.additional_actor_net = nn.Sequential(
            layer_init_orthogonal(nn.Linear(info['PartialState'].shape[0], 64), std=np.sqrt(2)),
            NonLinearity(),
        )
        self.n_parameters = count_parameters(self)

    def get_backbone(self, x):
        if hasattr(self, 'obs_rms'):
            x = (x - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
        if "Frame" in self.input_type:
            frame = x[0]
            assert frame.shape[1:] == self.check_shape, f"Expected frame shape {self.check_shape}, but got {frame.shape[1:]}"
            state = x[1]
            out_frame = self.cnn_backbone(frame).squeeze()
            out_state = self.additional_actor_net(state)
            return torch.concat((out_frame.reshape(out_frame.shape[0], -1), out_state), dim=-1)
        else:
            x = self.cnn_backbone(x)
            x = x.view(x.size(0), -1)
            return x
    
    def get_screen(self, envs=None, screen=None):
        frame_cfg = getattr(self, "frame_cfg", None)
        return get_screen(screen=screen, envs = envs, frame_cfg=frame_cfg)

class Model_from_SB(nn.Module):
    def __init__(self, model_path, vecnorm_path=None, env = None, device = 'cpu'):
        super().__init__()
        from stable_baselines3.common.vec_env import VecNormalize, VecEnv, DummyVecEnv
        from stable_baselines3 import SAC

        self.sac_model = SAC.load(model_path, device=device)
        self.actor = self.sac_model.policy.actor
        self.input_shape = {'State': self.sac_model.observation_space.shape}
        if vecnorm_path is not None:
            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda: env])
            self.env = VecNormalize.load(vecnorm_path, env)
            self.mean_tensor = torch.tensor(self.env.obs_rms.mean, dtype=torch.float32, device=device)
            self.st_dev_tensor = torch.sqrt(torch.tensor(self.env.obs_rms.var, dtype=torch.float32, device=device))
            self.clip = self.env.clip_obs
        self.deterministic = False
        self.input_type = ["State"]
        self.action_type = "Continuous"

        self.act_low = torch.tensor(self.sac_model.action_space.low, dtype=torch.float32)
        self.act_high = torch.tensor(self.sac_model.action_space.high, dtype=torch.float32)


    def preprocess_obs(self, obs_raw):
        try:
            obs = (obs_raw - self.mean_tensor) / (self.st_dev_tensor + 1e-8)
            obs = obs.clamp(-self.clip, self.clip)
        except:
            print(obs_raw.device, self.mean_tensor.device, self.st_dev_tensor.device)
        return obs
    def forward(self, x):
        if (x == torch.zeros_like(x)).all():
            self.print = True
            print("Zero input detected")
        action, _, _ = self.get_action(x)
        return action
    
    def get_action(self, x, action=None):
        x = self.preprocess_obs(x)
        mu, log_std, _ = self.actor.get_action_dist_params(x)
            
        self.logits = torch.cat( (mu, log_std) , dim=-1)
        self.actor.action_dist.proba_distribution(mu, log_std)  
        if action is None:
            if self.deterministic:
                tanh_action = torch.tanh(mu)
            else:
                tanh_action =  self.actor.action_dist.sample() 
            action = (self.act_high - self.act_low) * (tanh_action + 1.0) / 2.0 + self.act_low
        else:
            tanh_action = 2 * (action - self.act_low) / (self.act_high - self.act_low) - 1
        
        log_prob = self.actor.action_dist.log_prob(tanh_action)  
        entropy = self.actor.action_dist.entropy()
        if entropy is None:
            entropy = log_prob
        return action, log_prob, entropy
    
    def get_logits(self):
        return self.logits
    def create_distribution_from_logits(self, logits):
        mu, log_std = torch.chunk(logits, 2, dim=-1)
        distribution = Normal(mu, log_std.exp())  # Distribution for KL computation
        return distribution

    def to(self, device):
        self.device = device
        self.sac_model.policy.to(device)
        self.mean_tensor = self.mean_tensor.to(device)
        self.st_dev_tensor = self.st_dev_tensor.to(device)
        self.act_low = self.act_low.to(device)
        self.act_high = self.act_high.to(device)
        return self
    
    def get_last_distribution(self):
        mu, log_std = torch.chunk(self.logits, 2, dim=-1)
        return Normal(mu, log_std.exp())


def get_dataset(size = '100k'):
    if size == '100k':
        dataset = torch.load("./Results/Pusher/Teacher/dataset_100000steps.pt", weights_only=False, map_location=torch.device('cpu'))
    elif size == '50k':
        dataset = torch.load("./Results/Pusher/Teacher/dataset_50000steps.pt", weights_only=False, map_location=torch.device('cpu'))
    elif size == '10k':
        dataset = torch.load("./Results/Pusher/Teacher/dataset_10000steps.pt", weights_only=False, map_location=torch.device('cpu'))
    elif size == '5k':
        dataset = torch.load("./Results/Pusher/Teacher/dataset_5000steps.pt", weights_only=False, map_location=torch.device('cpu'))    
    else:
        raise ValueError("Invalid dataset size")
    return dataset

class Model_from_SB(nn.Module):
    def __init__(self, model_path, vecnorm_path=None, env = None, device = 'cpu'):
        super().__init__()
        from stable_baselines3.common.vec_env import VecNormalize, VecEnv, DummyVecEnv
        from stable_baselines3 import SAC

        self.sac_model = SAC.load(model_path, device=device)
        self.actor = self.sac_model.policy.actor
        self.input_shape = {'State': self.sac_model.observation_space.shape}
        if vecnorm_path is not None:
            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda: env])
            self.env = VecNormalize.load(vecnorm_path, env)
            self.mean_tensor = torch.tensor(self.env.obs_rms.mean, dtype=torch.float32, device=device)
            self.st_dev_tensor = torch.sqrt(torch.tensor(self.env.obs_rms.var, dtype=torch.float32, device=device))
            self.clip = self.env.clip_obs
        self.deterministic = False
        self.input_type = ["State"]
        self.action_type = "Continuous"

        self.act_low = torch.tensor(self.sac_model.action_space.low, dtype=torch.float32)
        self.act_high = torch.tensor(self.sac_model.action_space.high, dtype=torch.float32)


    def preprocess_obs(self, obs_raw):
        obs = (obs_raw - self.mean_tensor) / (self.st_dev_tensor + 1e-8)
        obs = obs.clamp(-self.clip, self.clip)
        return obs
    def forward(self, x):
        if (x == torch.zeros_like(x)).all():
            self.print = True
            print("Zero input detected")
        action, _, _ = self.get_action(x)
        return action
    
    def get_action(self, x, action=None):
        x = self.preprocess_obs(x)
        mu, log_std, _ = self.actor.get_action_dist_params(x)
            
        self.logits = torch.cat( (mu, log_std) , dim=-1)
        self.actor.action_dist.proba_distribution(mu, log_std)  
        if action is None:
            if self.deterministic:
                tanh_action = torch.tanh(mu)
            else:
                tanh_action =  self.actor.action_dist.sample() 
            action = (self.act_high - self.act_low) * (tanh_action + 1.0) / 2.0 + self.act_low
        else:
            tanh_action = 2 * (action - self.act_low) / (self.act_high - self.act_low) - 1
        
        log_prob = self.actor.action_dist.log_prob(tanh_action)  
        entropy = self.actor.action_dist.entropy()
        if entropy == None:
            entropy = log_prob.mean(-1)  
        return action, log_prob, entropy
    
    def get_logits(self):
        return self.logits
    def create_distribution_from_logits(self, logits):
        mu, log_std = torch.chunk(logits, 2, dim=-1)
        distribution = Normal(mu, log_std.exp())  # Distribution for KL computation
        return distribution

    def to(self, device):
        self.device = device
        self.sac_model.policy.to(device)
        self.mean_tensor = self.mean_tensor.to(device)
        self.st_dev_tensor = self.st_dev_tensor.to(device)
        self.act_low = self.act_low.to(device)
        self.act_high = self.act_high.to(device)
        return self
        