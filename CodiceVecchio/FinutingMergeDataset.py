# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
import itertools
import torch.optim as optim
import wandb
import torch.nn.functional as F
from typing import Literal

from Utils_PPO import make_env, set_seeds, PPO_Agent, Image_model, CreateBuffer_fromEpisode
from My_env import LunarLander
from Test_model import test_model, TestModel_underTeacher


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    device: str = "cuda:7" if torch.cuda.is_available() else "cpu"

    # Algorithm specific arguments
    num_iterations: int = 300
    """total timesteps of the experiments"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 1024 # 128
    """the number of steps to run in each environment per policy rollout"""
    batch_size: int = 16
    """the number of mini-batches"""
    gaussian_noise_std: float = 0.0
    """the standard deviation of the gaussian noise added to the actions"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    path_folder: str = "./Merge_Dataset_Finetuning_Results/"
    """the path to the folder where the model and other files are saved"""
    distillation_loss: str = 'CE'
    which_model: Literal["best", "<300"] = "best"

    from_scrath : bool = False
    """whether to load the model from scratch or to load a pre-trained model"""
    # load_path: str = "/home/l.callisti/Distillation_LunarLander/PPO/TrainResult/Distillation_loss_KL_algorithm_PPO_student_Image_20250912_172350/student_model_distilled_KL_best_ID.pth"
    # """the path from which the pre-trained model is loaded"""


def FinetuningMergeDataset(Teacher, Student, memory, optimizer, args):
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, False, run_name, args.path_folder, all_render=True) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    envs.reset()

    Buffer = CreateBuffer_fromEpisode(memory)
    Buffer.CreateBuffer(num_frames = 4, skipped_frames=1)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    logprob_T_memory = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    # Start the game
    global_step = 0
    start_time = time.time()
    set_seeds(args.seed)
    next_obs, _ = envs.reset(seed=args.seed)
    envs.action_space.seed(args.seed)
    next_frame = Student.init_deque(Student.get_screen(envs))
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    Accuracy_during_collection = torch.zeros((args.num_steps, args.num_envs))

    frames_memory = torch.zeros((args.num_steps, args.num_envs) + next_frame.shape[1:]).to(args.device)

    test_env = LunarLander(render_mode='rgb_array', random_initial_state=True, enable_wind=False, apply_initial_random=False)
    seed_ID, seed_OD = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210], [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    TesterSAccury_OD = TestModel_underTeacher(Teacher, seed_OD, test_env)

    for iteration in range(1, args.num_iterations + 1):

        if iteration % 10 == 1:
            mean_rwd_OD, accuracy_OD_under_S = test_model(Student, test_env, 10, seed_OD, video_folder=args.path_folder, model_B=Teacher, title = 'OD', title_A='model', title_B='Teacher', rec = True, dicts = None, device = args.device)
            accuracy_OD_under_T = TesterSAccury_OD.TestModel(Student)

            wandb.log({"losses/Test-accucary_under_T": accuracy_OD_under_T, "losses/Test-accucary_under_S": accuracy_OD_under_S, "losses/Test-mean_rwd": mean_rwd_OD, "global_step": global_step, 'iteration': iteration})
            Student.save_model(-999, seed_ID, mean_rwd_OD, seed_OD, epoch = iteration, path = args.path_folder, title = str(iteration), wandb_bool = True)
            print(f"OD under T {accuracy_OD_under_T:.2f} - OD under S {accuracy_OD_under_S:.2f} - OD reward {mean_rwd_OD:.2f}")

            next_obs, _ = envs.reset()
            next_frame = Student.init_deque(Student.get_screen(envs))
            next_obs = torch.Tensor(next_obs).to(args.device)

        # Collect episodes
        if iteration < 51:
            Running_Average = -999
            number_of_episodes = 0
            current_reward = torch.zeros(args.num_envs).to(args.device)
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                frames_memory[step] = next_frame
                

                with torch.no_grad():
                    action, _= Student(next_frame.to(args.device))
                    action_T, _ = Teacher(next_obs.to(args.device))
                Accuracy_during_collection[step] = (action.cpu() == action_T.cpu())

                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                next_frame = Student.append_deque(Student.get_screen(envs), next_done)
                logprob_T_memory[step] = Teacher.get_logprobs()
                rewards[step] = torch.tensor(reward).to(args.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)

                current_reward += rewards[step]
                if next_done.sum() > 0:
                    Running_Average = ( Running_Average * number_of_episodes + ( current_reward * next_done ).sum() ) / (number_of_episodes + next_done.sum())
                number_of_episodes += next_done.sum()
                current_reward[next_done == 1] = 0

            if Running_Average == -999:
                Running_Average = current_reward.mean()
            wandb.log({"losses/Average_reward": Running_Average, "losses/Accuracy_during_collection": Accuracy_during_collection.mean(), "iteration": iteration})


            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_frames =  frames_memory.reshape((-1,) + next_frame.shape[1:])
            b_logprob_T = logprob_T_memory.reshape((-1, envs.single_action_space.n))

            selected_index = torch.randperm(Buffer.states_tensor.shape[0])[:b_obs.shape[0]]
            Buffer.states_tensor[selected_index] = b_obs.cpu()
            Buffer.buffer_tensor[selected_index] = b_frames.cpu()
            Buffer.actions_tensor[selected_index] = b_logprob_T.cpu()
            dataloader = Buffer.Dataloader(batch_size=args.batch_size, gaussian_noise_std=args.gaussian_noise_std)

        print(f"\nStarting finetuning iteration {iteration}/{args.num_iterations}...")

        for i in range(args.update_epochs):
            set_seeds(1000+i)
            epoch_start_time = time.time()
            loss_epoch = 0
            
            samples_processed = 0
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    if batch_idx == 0: 
                        print(f"Epoch {i+1}, processing batch {batch_idx+1}/{len(dataloader)}", end='\r')
                    else:
                        print(f"Epoch {i+1}, processing batch {batch_idx+1}/{len(dataloader)} - Time for last 100 batches: {time.time() - batch_start_time:.2f}s", end='\r')
                    batch_start_time = time.time()
                
                step += 1   
                
                logprob_T = batch[2].to(args.device, non_blocking=True)
                frames = batch[0].to(args.device, non_blocking=True)
                _, logit_actions_S = Student(frames)
                logprob_S = Student.get_logprobs()

                # Calcola la loss utilizzando il gestore centralizzato
                if args.distillation_loss == 'CE':
                    if logprob_T.dim() == 1:
                        logprob_T = logprob_T.unsqueeze(0)
                    mask_idx = torch.argmax(logprob_T, dim=1)
                    one_hot_mask = F.one_hot(mask_idx, num_classes=logprob_T.shape[1]).type_as(logprob_T)
                    loss = -(one_hot_mask * logprob_S).sum(dim=1).mean()
                elif args.distillation_loss == 'KL':
                    loss = F.kl_div(logprob_S, logprob_T.detach(), reduction='batchmean', log_target=True)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                samples_processed += logprob_T.size(0)
                loss_epoch += loss.detach() 

            
            epoch_time = time.time() - epoch_start_time
            loss_mean = (loss_epoch / len(dataloader)).item() 
            samples_per_sec = samples_processed / epoch_time

            wandb.log({
                "losses/Loss each epoch": loss_mean, 
                "iteration": iteration,
                "Stats/epoch_time_seconds": epoch_time,
                "Stats/samples_per_second": samples_per_sec,
                "Stats/learning_rate": optimizer.param_groups[0]['lr']
            })
            print(f"\nEpoch {i+1} completed in {epoch_time:.2f}s - Loss: {loss_mean:.6f} - Samples/sec: {samples_per_sec:.2f}")
        # Upload videos to wandb e poi elimina i file locali
        # if args.track and args.capture_video:
        #     import glob
        #     video_dir = os.path.join(args.path_folder, "videos", run_name)
        #     video_files = sorted(glob.glob(f"{video_dir}/*.mp4"))
        #     for video_path in video_files:
        #         wandb.log({"video": wandb.Video(video_path, caption=os.path.basename(video_path), format="mp4"), "iteration": iteration})
        #         os.remove(video_path)


    # envs.close()
    for env in envs.envs:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"lr_{args.learning_rate:.2e}_loss_{args.distillation_loss}_model_{args.which_model}"
   

    wandb.init(
        project="FT-merge-dataset",
        config=vars(args),
        name=run_name,
        # monitor_gym=True,
        save_code=True,
        settings=wandb.Settings(code_dir=".")
    )

    Teacher = PPO_Agent(envs=LunarLander()).to(args.device)
    weights = torch.load("/home/l.callisti/Distillation_LunarLander/PPO/Teacher_files/Teacher_lr_1.00e-04_ent_coef_1.00e-02_Anneal_ent_ppo_agent.pt", map_location=args.device, weights_only=False)
    Teacher.load_state_dict(weights)
    env = LunarLander(render_mode='rgb_array', random_initial_state=True, enable_wind=False, apply_initial_random=False)
    env.reset()

    if args.distillation_loss == 'KL':
        if args.which_model == 'best':
            args.load_path = "/home/l.callisti/Distillation_LunarLander/PPO/TrainResult/Distillation_loss_KL_algorithm_PPO_student_Image_20250912_172350/student_model_distilled_KL_best_ID.pth"
        elif args.which_model == '<300':
            args.load_path = "/home/l.callisti/Distillation_LunarLander/PPO/TrainResult/Distillation_loss_KL_algorithm_PPO_student_Image_20250912_172350/student_model_distilled_KL_epoch_241.pth"
    elif args.distillation_loss == 'CE':
        if args.which_model == 'best':
            args.load_path = "/home/l.callisti/Distillation_LunarLander/PPO/TrainResult/Distillation_loss_My_CE_algorithm_PPO_20250929_114303/student_model_distilled_My_CE_best_ID.pth"
        elif args.which_model == '<300':
            args.load_path = "/home/l.callisti/Distillation_LunarLander/PPO/TrainResult/Distillation_loss_My_CE_algorithm_PPO_20250929_114303/student_model_distilled_My_CE_epoch_261.pth"
    if args.from_scrath:
        Student = Image_model(env=env).to(args.device)
    else:
        Student = Image_model.load_model_from_path(env=env, path=args.load_path, info = True).to(args.device)
    if args.from_scrath:
        optimizer = optim.Adam(Student.parameters(), lr=args.learning_rate, eps=1e-5, betas = (0.9999, 0.9999))
        Student.train()
    else:
        for param in Student.CNN.parameters():
            param.requires_grad = False
        Student.CNN.eval()
        Student.fc2.train()
        optimizer = optim.Adam(Student.fc2.parameters(), lr=args.learning_rate, eps=1e-5)
        Student.eval()
    memory = torch.load("/home/l.callisti/Distillation_LunarLander/PPO/Datasets/PPO_LunarLander_Dataset_523ep_100153elt.pth", map_location='cpu')
    if not os.path.exists(args.path_folder):
        os.makedirs(args.path_folder)


    FinetuningMergeDataset(Teacher, Student, memory, optimizer, args)

'''
python FinutingMergeDataset.py --distillation-loss KL --device cuda:5 --learning_rate 1e-6; python FinutingMergeDataset.py --distillation-loss KL --device cuda:5 --learning_rate 1e-5 
python FinutingMergeDataset.py --distillation-loss CE --device cuda:7 --learning_rate 1e-6; python FinutingMergeDataset.py --distillation-loss CE --device cuda:7 --learning_rate 1e-5

'''