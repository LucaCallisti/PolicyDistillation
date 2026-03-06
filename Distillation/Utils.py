import re
from Algorithm.seeds import get_seeds_id, get_seeds_od


# def get_PPD_paper_config():
#     default_config_PPD_paper = {
#         "torch_deterministic": True,
#         "num_envs": 18,         
#         "num_steps": 64,        # 5 per T/S distillation
#         "num_iterations": 1737, # PPD change   (2 million frames / (18 envs * 64 steps) = 1736.11)
#         "learning_rate": 3e-4,  # PPD change
#         "gamma": 0.999,         # PPD change
#         "gae_lambda": 0.9,      # PPD change
#         "clip_coef": 0.3,
#         "clip_vloss": True,
#         "ent_coef": 0.0,        # PPD change
#         "vf_coef": 1.0,
#         "max_grad_norm": 0.5,
#         "update_epochs": 4,     # PPD change
#         "minibatch_size": 64,   # sarebbe 512
#         "norm_adv": True,
#         "anneal_lr": True,
#         "anneal_ent_coef": True,
#         'Test_frequency': 25,
#         "reset_after_test": True,
#         'Verbose_frequency': 25,
#         'PPD_coef': 1,        # PPD change # Da provare (0.5, 1, 2, 5)
#     }
#     return default_config_PPD_paper
NUM_ENVS = 18

def get_dict_envs(mode, folder_path, run_name, env_name, wrappers):
    IDseeds = get_seeds_id()
    ODseeds = get_seeds_od()
    
    dict_enviroment = {
        "run_name": run_name,
        "env_name": env_name,
        "render_idx": [],
        "record_video_idx": [],
        "folder_path" : folder_path,
        'wrappers': wrappers,
        'num_envs': NUM_ENVS
    }
    dict_test_enviroment = {
        "run_name": run_name,
        "env_name": env_name,
        "render_idx": list(range(0, len(IDseeds))),
        "record_video_idx": [0],
        "IDseeds": IDseeds,
        "ODseeds": ODseeds,
        "folder_path" : folder_path,
        'wrappers': wrappers
    }
    if 'Impaala' in mode:
        dict_enviroment.update({
            "render_idx": list(range(0, NUM_ENVS)),
            'wrappers': wrappers
        })
        dict_test_enviroment.update({
            "render_idx": list(range(0, len(IDseeds))),
            'wrappers': wrappers
        })
    return dict_enviroment, dict_test_enviroment



def get_base_config():
    config = {
        "num_envs": NUM_ENVS,
        "learning_rate": 3e-4,
        "ent_coef": 0.0,
        "gamma": 0.999,
        "gae_lambda": 0.9,
        "clip_coef": 0.3,
        "clip_vloss": True,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
        "norm_adv": True,
        "anneal_lr": False,
        "anneal_ent_coef": True,
        "Test_frequency": 25,
        "Verbose_frequency": 25,
        "minibatch_size": 256,
    }
    return config

def get_ppd_config(update_step=500000, PPD_coef=1):
    config = get_base_config()
    config["num_steps"] = 64
    num_iter = update_step // ( NUM_ENVS * config["num_steps"] * config['update_epochs'] // config["minibatch_size"] )

    config.update({
        "num_iterations": num_iter,
        "PPD_coef": PPD_coef,
    })
    return config

def get_distillation_config(update_step = 500000, distillation_type="Teacher"):
    config = get_base_config()
    config["num_steps"] = 5  
    num_iter = update_step // ( NUM_ENVS * config["num_steps"] * config['update_epochs'] // config["minibatch_size"] )

    config.update({
        "num_iterations": num_iter,  
        "distillation_type": distillation_type,
        'optimization_steps': update_step
    })
    return config


def parse_path(path):
    size, mode, alpha = None, None, None
    for s in ['5k', '10k', '50k', '100k']:
        if s in path:
            size = s
            break
    if size is None: raise ValueError("Size must be one of '5k', '10k', '50k', '100k'")
    for m in ['State', 'ImpaalaSmall', 'ImpaalaMid', 'ImpaalaBig', 'Impaala']:
        if m in path:
            mode = m
            break
    if mode is None: raise ValueError("Mode must be one of 'State', 'Impaala', 'ImpaalaSmall', 'ImpaalaMid', 'ImpaalaBig'")
    for a in [0.0, 0.6, 0.8]:
        if f"{a}" in path:
            alpha = a
            break
    if alpha is None: raise ValueError("Alpha must be one of '0.0', '0.6', '0.8'")
    match = re.search(r'run([0-5])', path)
    if match:
        run_index = int(match.group(1))
    else:
        raise ValueError("Run index must be one of 'run0', ..., 'run5'")
    return size, mode, alpha, run_index