import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
from Algorithm.PPD import PPD
from Algorithm.StudentTeacher_distillation import Student_Distillation, Teacher_Distillation
from Algorithm.CollectDataset import Collect_Dataset
from Distillation.Utils import get_dict_envs, get_distillation_config, get_ppd_config, parse_path
from Enviroment.My_wrapper import AddFrameObsWrapper, AddPartialStateObsWrapper
from Models.Utils import get_screen
from Algorithm.BehaviouralCloning import BehaviouralCloning
from Algorithm.seeds import get_model_seed
import warnings
warnings.filterwarnings("ignore", message="adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation")



from Models.Model_Pusher import get_Teacher_model, StudentModelPusher, _get_wrappers, get_dataset



import torch
import warnings 
warnings.filterwarnings("ignore", message="Overwriting existing videos")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")



def Train_student_PPD(mode = 'State', run_index = 0):
    Teacher = get_Teacher_model()
    Student = StudentModelPusher(mode=mode, critic_net=True, seed=get_model_seed(run_index))


    print("Student model device", args.device)
   
    default_config = get_ppd_config(PPD_coef=args.PPD_parameter, update_step=100_000)
    
    default_config.update({"run_name": f"PPDcoef{args.PPD_parameter}_{mode}", "PPD_coef": args.PPD_parameter, 'env_name': 'Pusher-v5'})
    folder_path = "./Results/Pusher/PPD/"
    dict_enviroment, dict_test_enviroment = get_dict_envs(mode, folder_path, wrappers = _get_wrappers(mode), run_name=default_config["run_name"], env_name=default_config["env_name"])
    PPD_trainer = PPD(
        Student=Student,
        Teacher=Teacher,
        path_folder=dict_enviroment["folder_path"],
        dict_enviroment=dict_enviroment,
        device=args.device,
        config=default_config,
        dict_test_enviroment=dict_test_enviroment,
        Async_env=False,
        run_index=run_index,
    )
    print("Starting PPD training...")
    PPD_trainer.train()


def Train_TeacherStudent_distillation(mode = 'State', run_index = 0):
    Teacher = get_Teacher_model()
    Student = StudentModelPusher(mode=mode, critic_net=True, seed=get_model_seed(run_index))

    if args.Tdistillation:
        string = "Teacher"
    else:
        string = "Student"

    default_config = get_distillation_config(update_step=100_000, distillation_type=string)
    default_config.update({
        "env_name": "Pusher-v5",
        "run_name": f"{string}Distillation_{mode}"})
    folder_path = "./Results/Pusher/StudentDistillation/"

    dict_enviroment, dict_test_enviroment = get_dict_envs(mode, folder_path, wrappers=_get_wrappers(mode), run_name=default_config["run_name"], env_name=default_config["env_name"])
    if args.Tdistillation:
        dist_fun = Teacher_Distillation
    elif args.Sdistillation:
        dist_fun = Student_Distillation
    Distillation_trainer = dist_fun(
            student=Student,
            teacher=Teacher,
            path_folder=dict_enviroment["folder_path"],
            dict_enviroment=dict_enviroment,
            device=args.device,
            config=default_config,
            dict_test_enviroment=dict_test_enviroment,
            run_index=run_index,
        )
    Distillation_trainer.train()


def Collect_data(num_data = 10000, device='cpu'):
    num_envs = 50
    Teacher = get_Teacher_model()
    path_folder = "./Results/Teachers/Pusher/"
    dict_enviroment, dict_test_enviroment = get_dict_envs('Impaala', path_folder, wrappers = _get_wrappers('Impaala'), run_name='CollectData', env_name='Pusher-v5')
    dict_enviroment.update({'render_idx': list(range(num_envs)), 'record_video_idx': []})

    frame_cfg = {
                "crop_index": (70, 230, 70, 410),
                "shape": (64, 128),
                "grayscale": True,
                "normalize": True,
            }
    print("Collecting data...", num_data)
    Collect_Dataset(
        model=Teacher,
        num_data=num_data,
        path_folder=path_folder,
        dict_enviroment=dict_enviroment,
        num_envs=num_envs,
        additional_wrapper=[AddFrameObsWrapper, AddPartialStateObsWrapper],
        # additional_obs_shape={'Frame': (150, 380), 'PartialState': (17,)},
        additional_obs_shape={'Frame': (64, 128), 'PartialState': (17,)},
        transform_frame=lambda x: get_screen(screen=x, frame_cfg=frame_cfg),
        device=device,
    )

def BC_phase(mode='Impaala', size = '100k', alpha = 0.0, run_index = 0):
    Teacher = get_Teacher_model().to(args.device)
    dataset = get_dataset(size = size)
    loss_type = args.loss_type
    path_folder = f"./Results/Pusher/OurAlgorithm1_{loss_type}/"
    Student = StudentModelPusher(mode=mode, seed=get_model_seed(run_index))
    dict_enviroment, dict_test_enviroment = get_dict_envs(mode, path_folder, wrappers = _get_wrappers(mode), run_name=f'{size}_alpha{alpha}_{loss_type}_{mode}', env_name='Pusher-v5')
    Alg = BehaviouralCloning(
        Student=Student,
        Teacher=Teacher,
        dataset=dataset,
        loss_type=loss_type,
        alpha=alpha,
        path_folder=path_folder,
        device=args.device,
        dict_enviroment=dict_enviroment,
        dict_test_enviroment=dict_test_enviroment,
        num_frames=4,
        skipped_frames=1,
        Async_env=False,
        run_index=run_index,
        args = args,
    )
    Alg.BC_phase(
        lr=3e-4,
        batch_size=64,
        optimization_steps=100000,
        gaussian_noise_std=0.0,
        test_every_n_steps=5000
    )

def Dagger_phase(path_folder):
    size, mode, alpha, run_index = parse_path(path_folder)
    args.mode = mode
    args.size = size

    Teacher = get_Teacher_model().to(args.device)
    dataset = get_dataset(size = size)
    loss_type = args.loss_type
    target_steps = [20_000, 50_000, 80_000]
    target_steps = [1_500, 5_000, 10_000, 20_000]

    # 1. Collect all models and their info
    model_infos = []
    for model in os.listdir(path_folder):
        if model.endswith(".pth") and 'best' not in model:
            model_path = os.path.join(path_folder, model)
            model_info = torch.load(model_path, map_location=args.device, weights_only=False)
            model_infos.append({
                'path': model_path,
                'epochs': model_info['epochs'],
                'optimization_steps': model_info['optimization_steps'],
                'total_optimization_steps': model_info['total_optimization_steps'],
                'model_info': model_info
            })

    # 2. Sort by optimization_steps
    model_infos.sort(key=lambda x: x['optimization_steps'])

    # 3. Select the first model that exceeds each target_steps
    selected = []
    used_targets = set()
    for t in target_steps:
        for m in model_infos:
            if m['optimization_steps'] >= t and t not in used_targets:
                selected.append(m)
                used_targets.add(t)
                break

    for m in selected:
        epoch_of_model = m['epochs']
        optimization_steps_of_model = m['optimization_steps']
        total_optimization_steps = m['total_optimization_steps']
        print(f"Selected model at epoch {epoch_of_model} with {optimization_steps_of_model} optimization steps for OurAlgorithm2 Phase Two.")

        Student = StudentModelPusher.load_from_dict(m['model_info'])
        dict_enviroment, dict_test_enviroment = get_dict_envs('Impaala', path_folder, wrappers = _get_wrappers('Impaala'), run_name=f'{size}_alpha{alpha}_{loss_type}_{mode}_Nstep{optimization_steps_of_model}_Alpha{args.mode_alpha}', env_name='Pusher-v5')
        num_envs = dict_enviroment['num_envs']

        # Introduce 5% of new data per rollout and run 3 epochs over the full dataset
        p = 0.05 # percentage of new data to introduce at each rollout
        epochs_for_rollout = 3
        if size == '5k':
            rollout_steps = 5000 * p
            update_steps_per_rollout = epochs_for_rollout * 5000 // 64 # number of steps to go through the whole dataset with batch size 64
        elif size == '10k':
            rollout_steps = 10000 * p 
            update_steps_per_rollout = epochs_for_rollout * 10000 // 64
        elif size == '50k':
            rollout_steps = 50000 * p
            update_steps_per_rollout = epochs_for_rollout * 50000 // 64
        elif size == '100k':
            rollout_steps = 100000 * p
            update_steps_per_rollout = epochs_for_rollout * 100000 // 64
        rollout_steps = 2500
        update_steps_per_rollout = 5000

        Alg = BehaviouralCloning(
            Student=Student,
            Teacher=Teacher,
            dataset=dataset,
            loss_type=loss_type,
            alpha = 0.0,
            path_folder=path_folder,
            device=args.device,
            dict_enviroment=dict_enviroment,
            dict_test_enviroment=dict_test_enviroment,
            num_frames=4,
            skipped_frames=1,
            Async_env=False,
            mode_alpha=args.mode_alpha,
            run_index=run_index,
            args = args,
        )

        Alg.Dagger_phase(
            lr=1e-5,
            batch_size=64,
            optimization_steps_to_do=total_optimization_steps // 2,
            optimization_steps_done=optimization_steps_of_model,
            update_steps_per_rollout=update_steps_per_rollout,
            rollout_steps=rollout_steps,
            gaussian_noise_std=0.0,
            test_every_n_steps=5000
        )
        del Alg



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--teacher', default = False, action='store_true', help='Train the teacher agent')
    argparser.add_argument('--PPD', default = False, action='store_true', help='Train the student agent with PPD')
    argparser.add_argument('--mode', type=str, choices=['State', 'Impaala', 'ImpaalaSmall', 'ImpaalaMid', 'ImpaalaBig'], default='State', help='Mode to use for PPD training')
    argparser.add_argument('--PPD_parameter', choices=[0.5, 1, 2, 5], default=5, type=float, help='PPD parameter to use (only for PPD training)')
    argparser.add_argument('--Sdistillation', default = False, action='store_true', help='Train the student agent with student distillation')
    argparser.add_argument('--Tdistillation', default = False, action='store_true', help='Train the student agent with teacher distillation')
    argparser.add_argument('--device', type=str, default = 'cuda:1', help='Device to use for training (e.g., cuda:0)')
    argparser.add_argument('--C_data', default = False, action='store_true', help='Collect data from the teacher agent')
    argparser.add_argument('--BC_phase', default = False, action='store_true', help='Train the student agent with BC_phase')
    argparser.add_argument('--Dagger_phase', default = False, action='store_true', help='Train the student agent with Dagger_phase')
    argparser.add_argument('--num_data', type=int, default = 100000, help='Number of data points to collect')
    argparser.add_argument('--dataset', type=str, choices=['5k', '10k', '50k', '100k'], default='100k', help='Dataset size for OurAlgorithm1')
    argparser.add_argument('--alpha', type=float, default=0.0, help='Alpha parameter for OurAlgorithm1')
    argparser.add_argument('--run_index', type=int, default=0, help='Run index for seed selection (0-5)')
    argparser.add_argument('--folder_Dagger', type=str, help='Folder path for Dagger')
    argparser.add_argument('--loss_type', type=str, choices=['KL', 'NLL'], default='NLL', help='Loss type for OurAlgorithm1')
    argparser.add_argument('--mode_alpha', type=str, choices=['linear', 'constant', 'dynamic_mean', 'dynamic_max'], default='constant', help='Mode for alpha annealing in OurAlgorithm2')
    args = argparser.parse_args()

    if args.PPD:
        Train_student_PPD(mode = args.mode, run_index = args.run_index)
    if args.Sdistillation or args.Tdistillation:
        Train_TeacherStudent_distillation(mode = args.mode, run_index = args.run_index)
    if args.C_data:
        Collect_data(num_data = args.num_data, device=args.device)
    if args.BC_phase:
        BC_phase(mode = args.mode, size = args.dataset, alpha = args.alpha, run_index = args.run_index)
        path_folder_algorithm2=f'/home/l.callisti/PolicyDistillation/Results/Pusher/OurAlgorithm1_NLL/Phase_One/${args.dataset}_alpha${args.alpha}_NLL_${args.mode}_run0'
    if args.Dagger_phase:
        Dagger_phase(path_folder=args.folder_algorithm2)

