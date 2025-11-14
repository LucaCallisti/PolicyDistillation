import os
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import wandb
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo

from Utils_PPO import (
    PPO_Agent, set_seeds, CreateBuffer_fromEpisode, Image_model, Image_model_2head
)
from Test_model import Test_model_manager
from My_env import LunarLander


def get_title(args): 
    title = args.algorithm
    title += f'LR_{args.lr}_{args.distilation_loss}'
    return title

class DistillationLossManager:
    def __init__(self, algorithm='PPO'):
        self.algorithm = algorithm
        
    def KL_loss(self, log_prob_S, log_prob_T):
        """
        Kullback-Leibler Divergence Loss.
        
        Args:
            log_prob_S: Log probabilities dello studente
            log_prob_T: Log probabilities del teacher
        
        Returns:
            KL divergence loss
        """
        return F.kl_div(log_prob_S, log_prob_T.detach(), reduction='batchmean', log_target=True)

    def my_ce_loss(self, logprob_S, logit_T):
        """
        Custom Cross-Entropy Loss (esempio).
        
        Args:
            logprob_S: Log probabilities dello studente
            logit_T: Logits del teacher
        
        Returns:
            Custom cross-entropy loss
        """
        if logit_T.dim() == 1:
            logit_T = logit_T.unsqueeze(0)
        mask_idx = torch.argmax(logit_T, dim=1)
        one_hot_mask = F.one_hot(mask_idx, num_classes=logit_T.shape[1]).type_as(logit_T)
        ce_loss = -(one_hot_mask * logprob_S).sum(dim=1).mean()
        return ce_loss


    def compute_loss(self, loss_type, student_model=None, log_prob_S=None, logit_S=None, 
                    log_prob_T=None):
        """
        Metodo generale per calcolare qualsiasi tipo di loss.
        
        Args:
            loss_type: Tipo di loss da calcolare
            student_model: Modello studente (per loss combinate)
            log_prob_S: Log probabilities dello studente
            logit_S: Logits dello studente
            log_prob_T: Log probabilities del teacher
            alpha: Parametro alpha per soft CE
            beta: Parametro beta per CE scaling
        
        Returns:
            Computed loss
        """
        if loss_type == 'KL':
            return self.KL_loss(log_prob_S, log_prob_T)
        elif loss_type == 'My_CE':
            return self.my_ce_loss(log_prob_S, log_prob_T)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")



def distillation_train(memory, batch_size, epochs, lr):
    eval_env = LunarLander(render_mode='rgb_array', random_initial_state=True, enable_wind=False, apply_initial_random=False)
    h, w = memory['frames'][0].shape
    
    # Inizializza il gestore delle loss
    loss_manager = DistillationLossManager(algorithm=args.algorithm)    
    model_weights = torch.load("/home/l.callisti/Distillation_LunarLander/PPO/Teacher_files/Teacher_lr_1.00e-04_ent_coef_1.00e-02_Anneal_ent_ppo_agent.pt", map_location=device, weights_only=True)
    model_T = PPO_Agent(eval_env).to(device)
    model_T.load_state_dict(model_weights)
    model_T.eval()

    eval_env.reset()

    print("Creating buffer and DataLoader...")
    Buffer = CreateBuffer_fromEpisode(memory)
    Buffer.CreateBuffer(num_frames = args.num_frames, skipped_frames=args.skipped_frames)
    dataloader = Buffer.Dataloader(batch_size=batch_size, gaussian_noise_std=args.gaussian_noise_std)

    # Log buffer info to wandb
    wandb.log({
        "Params/total_buffer_sequences": len(Buffer.buffer_tensor),
        "Params/batches_per_epoch": len(dataloader)
    })

    epoch_done = 0
    if args.resume:
        Student_model = Image_model.load_model_from_path(path = args.checkpoint_path, env = eval_env, device=device)
        params = torch.load(args.checkpoint_path, weights_only=False)
        epoch_done = params['epoch']
    else:
        Student_model = Image_model(eval_env, num_frames=args.num_frames, skipped_frames=args.skipped_frames, p_dropout = 0).to(device)
        print('Student input size:', args.num_frames, h, w)
        summary(Student_model, input_size=(1, args.num_frames, h, w), device=str(device))  
    
    optimizer = torch.optim.Adam(Student_model.parameters(), lr=lr)

    step = 0

    # Profiling info
    total_samples = len(Buffer.buffer_tensor)
    
    print(f"Starting training:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Epochs: {epochs}")

    # Progress bar per le epoch
    epoch_pbar = tqdm(range(epoch_done, epochs), desc="Training", position=0)
 
    od_seeds = list(range(100, 110))
    TestModel = Test_model_manager(Teacher=model_T, seed_OD=od_seeds, path_ID=args.ID_dataset, env=eval_env, device=device)
    current_best_model_dict = {
        'model_dict': None,
        'mean_rwd_ID': -999,
        'mean_rwd_OD': -999,
        'epoch': None
    }
    aux = 1

    for i in epoch_pbar:
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
            
            logit_actions_T = batch[2].to(device, non_blocking=True)
            frames = batch[0].to(device, non_blocking=True)
            _, logit_actions_S = Student_model(frames)
            logprob_S = Student_model.get_logprobs()

            # Calcola la loss utilizzando il gestore centralizzato
            loss = loss_manager.compute_loss(
                loss_type=args.distilation_loss,
                log_prob_S=logprob_S,
                logit_S=logit_actions_S,
                log_prob_T=logit_actions_T,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            samples_processed += logit_actions_T.size(0)
            loss_epoch += loss.detach() 

        
        epoch_time = time.time() - epoch_start_time
        loss_mean = (loss_epoch / len(dataloader)).item() 
        samples_per_sec = samples_processed / epoch_time

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Loss': f'{loss_mean:.4f}',
            'Time': f'{epoch_time:.1f}s',
            'Samples/s': f'{samples_per_sec:.1f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        wandb.log({
            "loss/Loss each epoch": loss_mean, 
            "epoch": i + 1,
            "Stats/epoch_time_seconds": epoch_time,
            "Stats/samples_per_second": samples_per_sec,
            "Stats/learning_rate": optimizer.param_groups[0]['lr']
        })

        tqdm.write(f"Epoch {i + 1}/{epochs}, Loss: {loss_mean:.4f}, Time: {epoch_time:.1f}s, Samples/s: {samples_per_sec:.1f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if i % 20 == 0:
            tqdm.write(f"Running evaluation at epoch {i+1}...")
            eval_start_time = time.time()

            res = TestModel.test(Student_model)

            eval_time = time.time() - eval_start_time
            tqdm.write(f"Evaluation completed in {eval_time:.1f}s - ID: {res['mean_rwd_ID']:.2f}, OD: {res['mean_rwd_OD']:.2f}")

            wandb.log({
                f"loss/Avg reward test episodes ID": res['mean_rwd_ID'],
                f"loss/Avg reward test episodes OD": res['mean_rwd_OD'],
                "epoch": i + 1
            })

            if res['mean_rwd_ID'] > current_best_model_dict['mean_rwd_ID']:
                current_best_model_dict['model_dict'] = Student_model.state_dict().copy()
                current_best_model_dict['mean_rwd_ID'] = res['mean_rwd_ID']
                current_best_model_dict['mean_rwd_OD'] = res['mean_rwd_OD']
                current_best_model_dict['epoch'] = i + 1
                tqdm.write(f"New best model found at epoch {i+1} with ID reward {res['mean_rwd_ID']:.2f} and OD reward {res['mean_rwd_OD']:.2f}. Saving model...")
            if i >= aux * 100:
                aux += 1
                Student_model.save_model(mean_ID_reward=current_best_model_dict['mean_rwd_ID'], seed_Id=memory['seed_run'][:10], mean_OD_reward=current_best_model_dict['mean_rwd_OD'], seed_OD=od_seeds, epoch = i+1, path=args.path, title=f"{args.distilation_loss}_best<{i+1}", wandb_bool=True)
            
            Student_model.save_model(mean_ID_reward=res['mean_rwd_ID'], seed_Id=memory['seed_run'][:10], mean_OD_reward=res['mean_rwd_OD'], seed_OD=od_seeds, epoch = i+1, path=args.path, title=f"{args.distilation_loss}_epoch_{i+1}", wandb_bool=True)
            Student_model.train()
        
    
    epoch_pbar.close()
    
    # Training summary
    total_training_time = time.time() - epoch_pbar.start_t if hasattr(epoch_pbar, 'start_t') else 0
    tqdm.write(f"\n=== Training Completed ===")
    tqdm.write(f"Total epochs: {epochs}")
    tqdm.write(f"Best ID reward: {current_best_model_dict['mean_rwd_ID']:.2f}")
    tqdm.write(f"Best OD reward: {current_best_model_dict['mean_rwd_OD'] :.2f}")
    if total_training_time > 0:
        tqdm.write(f"Total training time: {total_training_time/3600:.2f} hours")
        tqdm.write(f"Average time per epoch: {total_training_time/epochs:.1f} seconds")
    
    # Log final summary to wandb
    wandb.log({
        "loss/final_best_ID_reward": current_best_model_dict['mean_rwd_ID'],
        "loss/final_best_OD_reward": current_best_model_dict['mean_rwd_OD'],
        "Params/total_training_epochs": epochs 
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarLander Collector")
    parser.add_argument('--epochs', type=int, default=1001)
    parser.add_argument('--batch-size', type=int, default=64) 
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--distilation-loss', type=str, default='My_CE', choices=['My_CE', 'KL'], help='Type of distillation loss to use')
    parser.add_argument('--memory-path', type=str, default="/home/l.callisti/Distillation_LunarLander/PPO/Datasets/PPO_LunarLander_Dataset_523ep_100153elt.pth")

    parser.add_argument('--num-frames', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--skipped-frames', type=int, default=1, help='Number of frames to skip for stacking')
    parser.add_argument('--gaussian-noise-std', type=float, default=0.01, help='Standard deviation of Gaussian noise added to input frames')

    parser.add_argument('--gpu', type=int, default=7, help='GPU id to use (default: 0)')
    parser.add_argument('--path', type=str, default="/home/l.callisti/Distillation_LunarLander/PPO/TrainResult")
    
    # To resume a rtaining from checkpoint
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--wandb-id', type=str, default='oan46scm', help='Wandb run ID to resume (required if --resume is used)')
    parser.add_argument('--checkpoint-path', type=str, default='/home/l.callisti/Distillation_LunarLander/PPO/TrainResult/Distillation_loss_1_algorithm_PPO_student_Image_20250915_173543/student_model_distilled_1_epoch_801.pth', help='Path to checkpoint file to resume from')
    
    args = parser.parse_args()    

    if args.resume:
        if args.wandb_id is None or args.checkpoint_path is None:
            raise ValueError("When using --resume, both --wandb-id and --checkpoint-path must be provided.")
        print(f"Resuming training from checkpoint: {args.checkpoint_path} with wandb ID: {args.wandb_id}")
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
        args.path = os.path.dirname(args.checkpoint_path)
    else:
        args.path = os.path.join(args.path, f"Distillation_loss_{args.distilation_loss}_algorithm_{args.algorithm}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(args.path, exist_ok=True)

    # Configura directory temporanee
    temp_dirs = {
        'wandb_cache': os.path.join(args.path, "wandb_cache"),
        'wandb_data': os.path.join(args.path, "wandb_data"), 
        'temp_dir': os.path.join(args.path, "temp")
    }
    # Imposta variabili d'ambiente
    os.environ["WANDB_CACHE_DIR"] = temp_dirs['wandb_cache']
    os.environ["WANDB_DATA_DIR"] = temp_dirs['wandb_data']
    os.environ["TMPDIR"] = temp_dirs['temp_dir']
    os.environ["TMP"] = temp_dirs['temp_dir']
    os.environ["TEMP"] = temp_dirs['temp_dir']

    for dir_path in temp_dirs.values():
        os.makedirs(dir_path, exist_ok=True)


    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    memory = torch.load(args.memory_path, map_location='cpu')

    if args.resume:
        # Configure wandb for resume or new run
        if '_'+args.distilation_loss not in args.checkpoint_path:
            raise ValueError("The distilation_loss in the checkpoint path does not match the current argument.")
        wandb.init(
        project="PPO-Distillation_v2",
        config=vars(args),
        name=get_title(args),
        dir=args.path,
        id=args.wandb_id,
        resume="must",
        settings=wandb.Settings(code_dir=".")
        )
    else:
        wandb.init(
        project="PPO-Distillation_v2",
        config=vars(args),
        name=get_title(args),
        dir=args.path,
        settings=wandb.Settings(code_dir="."),
        save_code=True
        )
    
    set_seeds(2)
    distillation_train(memory, args.batch_size, args.epochs, lr=args.lr)


'''
python Teacher2Student_v2.py --distilation-loss KL --gpu 1
python Teacher2Student_v2.py --distilation-loss CE --gpu 3
python Teacher2Student_v2.py --distilation-loss MSE --gpu 4
python Teacher2Student_v2.py --distilation-loss CE-Entropy --gpu 2
'''