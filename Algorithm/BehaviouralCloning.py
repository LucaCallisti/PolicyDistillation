import time
import gymnasium as gym
from Enviroment.Utils import make_env
from Algorithm.Test_model import test_model, TestModel_underTeacher
from Algorithm.Utils import get_input, CreateDataloaderFromDataset, set_seeds, get_accuracy
from Algorithm.seeds import SEEDS_ID, SEEDS_OD, get_phase_two_seeds, get_phase_one_seed_base
import torch
import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
class loss_functions:
    def __init__(self, loss_type: str, device: str = 'cpu'):
        assert loss_type in ['CE', 'MSE', 'KL', 'NLL'], "Invalid loss type. Choose from 'CE', 'MSE', 'KL', 'NLL'."
        self.loss_type = loss_type
        self.device = device
    
    def compute_loss(self, actions_S, actions_T, Student, Teacher, logits_T = None, reduction = 'mean'):
        if self.loss_type == 'CE':
            logits_S = Student.get_logits()
            loss = torch.nn.functional.cross_entropy(logits_S, actions_T.to(self.device, non_blocking=True).long(), reduction=reduction)
        elif self.loss_type == 'MSE':
            loss = torch.nn.functional.mse_loss(actions_S, actions_T.to(self.device, non_blocking=True), reduction=reduction)
        elif self.loss_type == 'KL':
            distribution_S = Student.get_last_distribution()
            if logits_T is None:
                distribution_T = Teacher.get_last_distribution()
            else:
                distribution_T = Teacher.create_distribution_from_logits(logits_T.to(self.device, non_blocking=True))
            loss = torch.distributions.kl.kl_divergence(distribution_T, distribution_S)
            if loss.dim() > 1:
                loss = loss.sum(dim=-1)  # Sum over action dimensions

            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'none':
                pass  # keep the loss as is
        elif self.loss_type == 'NLL':
            logits_S = Student.get_logits()
            mean_S, log_std_S = logits_S.chunk(2, dim=-1)

            low, high = Student.act_low.to(self.device), Student.act_high.to(self.device)
            tanh_action = 2 * (actions_T.to(self.device) - low) / (high - low) - 1 
            raw_action = torch.atanh(tanh_action.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) 

            dist_S = torch.distributions.Normal(mean_S, log_std_S.exp())
            loss = -dist_S.log_prob(raw_action).sum(dim=-1)  # Negative log-likelihood
            loss = loss + ((-Student.min_log_std - 0.5*torch.log(torch.tensor(2 * torch.pi)).to(self.device)) * actions_S.shape[-1] ).to(self.device)  # Add constant term to make loss non-negative

            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'none':
                pass  # keep the loss as is
        return loss

class BehaviouralCloning:
    def __init__(self, Student: object, Teacher: object, dataset, loss_type: str, alpha: float, 
                 path_folder: str, device: str, dict_enviroment: dict, dict_test_enviroment: dict, 
                 num_frames, skipped_frames, Async_env: bool = True, mode_alpha: str = 'constant', run_index: int = 0, args = None):
        
        # Core attributes
        self.Student = Student.to(device)
        self.Teacher = Teacher.to(device)
        self.alpha = alpha
        self.loss_type = loss_type
        self.dataset = dataset
        self.device = device
        self.run_index = run_index
        self.args = args

        # Environment config
        self.run_name = dict_test_enviroment["run_name"] + f"_run{run_index}"
        self.env_name = dict_test_enviroment["env_name"]
        self.wrapper = dict_test_enviroment['wrappers']
        self.loss_fn = loss_functions(loss_type, device=device)
        
        # Setup
        self.path_folder = path_folder
        self._setup_vector_env_function(Async_env)
        self._setup_environments(dict_enviroment, dict_test_enviroment)
        self._setup_buffer(num_frames, skipped_frames, mode_alpha)
        
        # Test checkpoints
        self.test_checkpoint_1phase_steps = []  # Track when tests were run (in optimization steps)
        
        self.df_info = pd.DataFrame()


    def _setup_paths(self, phase):
        """Create the necessary directories to save results."""
        os.makedirs(self.path_folder, exist_ok=True)
        if phase == 'BC':
            self.first_phase_folder = os.path.join(self.path_folder, 'BC', self.run_name)
            os.makedirs(self.first_phase_folder, exist_ok=True)
        elif phase == 'Dagger':
            self.second_phase_folder = os.path.join(self.path_folder, 'Dagger', self.run_name)
            os.makedirs(self.second_phase_folder, exist_ok=True)
        else:
            raise ValueError("Invalid phase. Choose 'BC' or 'Dagger'.")

    def _setup_vector_env_function(self, async_env: bool):
        """Select the function to create vectorized environments."""
        if async_env:
            self.vec_env_fun = gym.vector.AsyncVectorEnv
        else:
            self.vec_env_fun = gym.vector.SyncVectorEnv

    def _setup_environments(self, dict_enviroment: dict, dict_test_enviroment: dict):
        """Initialize training and test environments."""

        if dict_enviroment is not None:
            self.dict_enviroment = dict_enviroment
            self.dict_enviroment['IDseeds'] = SEEDS_ID
            self.dict_enviroment['ODseeds'] = SEEDS_OD
            self.envs = self._create_vector_env(dict_enviroment, autoreset=True)
        
        if dict_test_enviroment is not None:
            self.dict_test_enviroment = dict_test_enviroment
            self.dict_test_enviroment['IDseeds'] = SEEDS_ID
            self.dict_test_enviroment['ODseeds'] = SEEDS_OD
            self.test_envs = self._create_vector_env(dict_test_enviroment, autoreset=False)
            self._setup_testers(dict_test_enviroment)

    def _create_vector_env(self, env_config: dict, autoreset: bool = True):
        """Create a vectorized environment."""
        env_fns = [make_env(env_config, idx, wrappers=self.wrapper) for idx in range(len(env_config['IDseeds']))]
        if autoreset:
            return self.vec_env_fun(env_fns)
        else:
            return self.vec_env_fun(env_fns, autoreset_mode='Disabled')

    def _setup_testers(self, dict_test_enviroment: dict):
        """Initialize testers to evaluate the student under the teacher."""
        self.Tester_under_teacher_ID = TestModel_underTeacher(
            Teacher=self.Teacher, Student=self.Student, 
            seeds=dict_test_enviroment['IDseeds'], envs=self.test_envs
        )
        self.Tester_under_teacher_OD = TestModel_underTeacher(
            Teacher=self.Teacher, Student=self.Student, 
            seeds=dict_test_enviroment['ODseeds'], envs=self.test_envs
        )

    def _setup_buffer(self, num_frames: int, skipped_frames: int, mode_alpha: str):
        """Initialize and preprocess the data buffer."""
        self.Buffer = CreateDataloaderFromDataset(self.dataset, alpha=self.alpha, mode_alpha=mode_alpha)
        self.Buffer.PreprocessFrames(num_frames=num_frames, skipped_frames=skipped_frames)
    
    def BC_phase(self, lr, batch_size, optimization_steps, gaussian_noise_std, test_every_n_steps=1000):
        """
        Args:
            optimization_steps: Total number of gradient updates (batch steps) to perform.
            test_every_n_steps: Run evaluation every this many optimization steps.
        """
        print("Creating buffer and DataLoader...")
        dataloader = self.Buffer.GetDataloader(batch_size=batch_size, gaussian_noise_std=gaussian_noise_std)
        self._setup_paths('BC')
        wandb.init(
            entity='Distillation_RL',
            project=f"{self.env_name}_alpha{self.alpha}",
            name=f"BC_{self.run_name}",
            save_code=True,
            config = self.args
        )
        print(self.Student)
        print(f"Total parameters in Student: {self.Student.n_parameters:,}")

        optimizer = torch.optim.AdamW(self.Student.parameters(), lr=lr, weight_decay=1e-4)

        best_ID_reward = -1000
        best_OD_reward = -1000

        total_samples = len(self.Buffer)
        batches_per_epoch = len(dataloader)
        epochs_needed = -(-optimization_steps // batches_per_epoch)  # ceil division
    
        print(f"Starting training:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {batches_per_epoch}")
        print(f"  Target optimization steps: {optimization_steps:,}")
        print(f"  Epochs needed (approx): {epochs_needed}")
        print(f"  Test every: {test_every_n_steps} steps")

        self.global_step = 0  # optimization steps (batch updates)
        self.samples_seen = 0  # total samples seen
        self.i = 0  # epoch counter
        self.optimization_steps = optimization_steps

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimization_steps, eta_min=1e-6)
        test_optimized_step = list(range(0, test_every_n_steps, 1000)) + list(range(test_every_n_steps, optimization_steps, test_every_n_steps))
        print(f"Test steps: {test_optimized_step}")
        
        while self.global_step < optimization_steps:
            set_seeds(get_phase_one_seed_base(self.run_index) + self.i)
            self.Student.train()
            self.Student.deterministic = False

            if self.alpha > 0.0 and self.i > 0:
                self.Buffer.annealing_beta(current_iteration=self.global_step, total_iterations=optimization_steps)
                dataloader = self.Buffer.GetDataloader(batch_size=batch_size, gaussian_noise_std=gaussian_noise_std)
            

            loss_mean, mean_accuracy, epoch_time, samples_per_sec, batch_samples = self._update_step(
                dataloader, optimizer, max_steps=optimization_steps - self.global_step
            )
            self.samples_seen += batch_samples

            # Calcola log_std per monitoring
            last_logits_S = self.Student.get_logits()
            last_shape = last_logits_S.shape[-1]
            log_sigma = last_logits_S[..., last_shape//2:]
            
            # Buffer metrics (if using prioritized replay)
            buffer_metrics = self.Buffer.compute_buffer_metrics() if self.alpha > 0.0 else {}
            
            infos_to_log = {
                # X-axis metrics (use any of these in wandb)
                "x_axis/optimization_steps": self.global_step,
                "x_axis/samples_seen": self.samples_seen,
                "x_axis/epochs": self.i + 1,
                # Training metrics
                "Training/Loss": loss_mean, 
                "Training/Accuracy": mean_accuracy,
                "Training/learning_rate": optimizer.param_groups[0]['lr'],
                "Training/log_std_mean": log_sigma.mean().item(),
                # Stats
                "Stats/epoch_time_seconds": epoch_time,
                "Stats/samples_per_second": samples_per_sec,
            } | buffer_metrics
            
            # Test based on optimization steps instead of epochs
            if (len(test_optimized_step) > 0 and self.global_step >= test_optimized_step[0]) or self.global_step >= optimization_steps:
                test_optimized_step = [x for x in test_optimized_step if x > self.global_step]
                print(f"\nRunning evaluation at step {self.global_step:,} (epoch {self.i+1})...")

                eval_start_time = time.time()
                mean_reward_ID, accuracy_ID, mean_reward_OD, accuracy_OD, infos = self.test()
                eval_time = time.time() - eval_start_time
                print(f"Evaluation completed in {eval_time:.1f}s - ID: {mean_reward_ID:.2f}, OD: {mean_reward_OD:.2f}")

                infos_to_log = infos_to_log | infos
                
                # Generalization gaps
                infos_to_log["Generalization/ID_OD_reward_gap"] = mean_reward_ID - mean_reward_OD
                infos_to_log["Generalization/ID_OD_accuracy_gap"] = accuracy_ID - accuracy_OD

                if mean_reward_ID > best_ID_reward:
                    best_ID_reward = mean_reward_ID
                    print(f"New best ID reward: {best_ID_reward:.2f} at step {self.global_step:,}")
                    self.save_model(mean_ID_reward=mean_reward_ID, seed_ID=self.dict_test_enviroment['IDseeds'], mean_reward_OD=mean_reward_OD, seed_OD=self.dict_test_enviroment['ODseeds'], path=self.first_phase_folder, title=f"{self.loss_type}_BestModel_best_ID", wandb_bool=True)
            
                self.save_model(mean_ID_reward=mean_reward_ID, seed_ID=self.dict_test_enviroment['IDseeds'], mean_reward_OD=mean_reward_OD, seed_OD=self.dict_test_enviroment['ODseeds'], path=self.first_phase_folder, title=f"{self.loss_type}_step_{self.global_step}", wandb_bool=True)
                self.Student.train()
            
            self._load_infos(infos_to_log)
            self.i += 1
        
        # Log final summary to wandb
        wandb.log({
            "Training/final_best_ID_reward": best_ID_reward,
            "Training/final_best_OD_reward": best_OD_reward,
            "Params/total_optimization_steps": self.global_step,
            "Params/total_epochs": self.i + 1,
            "Params/total_samples": total_samples,
        })
        wandb.finish()

    def Dagger_phase(self, lr, batch_size, optimization_steps_to_do, optimization_steps_done, update_steps_per_rollout, rollout_steps, gaussian_noise_std=0, test_every_n_steps=5000):
        """
        Args:
            optimization_steps_to_do: Total number of gradient updates to perform.
            update_steps_per_rollout: Number of optimization steps after each rollout.
            rollout_steps: Number of rollout steps to collect before each update.
            test_every_n_steps: Run evaluation every this many optimization steps.
        """
        self._setup_paths('Dagger')
        wandb.init(
            entity='Distillation_RL',
            project=f"{self.env_name}_alpha{self.alpha}",
            name=f"Dagger_{self.run_name}",
            save_code=True,
            config = self.args
        )

        self.global_step = 0  # optimization steps (batch updates) in this phase
        self.samples_seen = 0  # total samples seen
        self.samples_collected = 0  # samples collected from rollouts
        self.step_to_be_done_in_phase_two = optimization_steps_to_do - optimization_steps_done
        self.test_checkpoint_2phase_steps = []
        best_ID_reward = -1000


        num_step_per_env = int(-(-rollout_steps // self.envs.num_envs))
        rollout_steps = int(num_step_per_env * self.envs.num_envs)  # Adjust to be multiple of num_envs

        self.next_obs, self.info = self.envs.reset(seed=get_phase_two_seeds(self.run_index)[0])
        self.next_obs = torch.tensor(self.next_obs, device=self.device)
        self.next_done = torch.ones(8, device=self.device)

        self.b_new_observations = self.Buffer.observation.copy()
        for k in self.b_new_observations.keys():
            self.b_new_observations[k] = self.b_new_observations[k][:rollout_steps]
        self.b_actions_T = self.Buffer.actions[:rollout_steps].clone()
        self.b_logits_distribution_T = self.Buffer.logits_distribution[:rollout_steps].clone()
        self.b_loss = torch.zeros(self.b_actions_T.shape[0], device=self.device)

        if "Frame" in self.Student.input_type:
            self.Student.reset_input_manager()
        if "Frame" in self.Teacher.input_type:
            self.Teacher.reset_input_manager()

        self.i = 0  # rollout counter

        print(f"Starting Phase 2 training:")
        print(f"  Target optimization steps (absolute): {optimization_steps_to_do:,}")
        print(f"  Already completed steps: {optimization_steps_done:,}")
        print(f"  Update steps per rollout: {update_steps_per_rollout}")
        print(f"  Rollout steps: {rollout_steps}")
        print(f"  Test every: {test_every_n_steps} steps")

        def get_total_optimization_steps():
            return optimization_steps_done + self.global_step
        optimizer = torch.optim.Adam(self.Student.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimization_steps_to_do-optimization_steps_done, eta_min=1e-6)
        test_optimized_step = list(range(0, test_every_n_steps, 1000)) + list(range(test_every_n_steps, self.step_to_be_done_in_phase_two, test_every_n_steps))

        # update error of the buffer with current loss
        if self.alpha > 0.0:
            with torch.no_grad():
                Student_keys = self.Student.input_type
                sequentialDataloader = self.Buffer.GetSequentialDataloader(batch_size=batch_size, gaussian_noise_std=gaussian_noise_std)
                for batch in sequentialDataloader:
                    obs_batch, actions_T_batch, logits_distribution_T_batch, weights, idxs, masks_old, maks_new = batch
                    if len(Student_keys) == 1:
                        input_S = obs_batch[Student_keys[0]].to(self.device, non_blocking=True)
                    else:
                        input_S = tuple([obs_batch[k].to(self.device, non_blocking=True) for k in Student_keys])
                    action_S, _, _ = self.Student.get_action(input_S)   
                    self.Buffer.update_errors(idxs, self.loss_fn.compute_loss(action_S, actions_T_batch, self.Student, self.Teacher, logits_distribution_T_batch, reduction = 'none').cpu())


        while get_total_optimization_steps() < optimization_steps_to_do:
            set_seeds(get_phase_two_seeds(self.run_index)[1] + self.i)
            self.Student.train()
            self.Student.deterministic = False
            if self.alpha > 0.0:
                self.Buffer.annealing_beta(current_iteration=self.global_step, total_iterations=self.step_to_be_done_in_phase_two)
            self._collect_rollouts(num_step_per_env)
            self.samples_collected += rollout_steps  # samples collected in this rollout

            self.Buffer.update_dataset(self.b_new_observations, self.b_actions_T, self.b_logits_distribution_T, self.b_loss, self.global_step, optimization_steps_to_do-optimization_steps_done)

            # Compute how many steps remain (relative to the absolute target)
            remaining_steps = optimization_steps_to_do - get_total_optimization_steps()
            steps_this_rollout = min(update_steps_per_rollout, remaining_steps)
            num_epochs = steps_this_rollout // int(-(-len(self.Buffer) // batch_size)) + 1
            steps_each_epoch = steps_this_rollout // num_epochs
            for epoch in range(num_epochs):
                # dataloader = self.Buffer.GetDataloader(batch_size=batch_size, gaussian_noise_std=gaussian_noise_std)
                dataloader = self.Buffer.get_balanced_datalaoder(batch_size=batch_size, gaussian_noise_std=gaussian_noise_std)
                loss_mean, mean_accuracy, epoch_time, samples_per_sec, batch_samples = self._update_step(
                    dataloader, optimizer, max_steps=steps_each_epoch
                )
                self.samples_seen += batch_samples

            # Buffer metrics (if using prioritized replay)
            buffer_metrics = self.Buffer.compute_buffer_metrics() if self.alpha > 0.0 else {}

            # Compute log_std for monitoring
            last_logits_S = self.Student.get_logits()
            last_shape = last_logits_S.shape[-1]
            log_sigma = last_logits_S[..., last_shape//2:]

            infos_to_log = {
                # X-axis metrics (use any of these in wandb)
                "x_axis/optimization_steps": get_total_optimization_steps(),
                "x_axis/samples_seen": self.samples_seen,
                "x_axis/samples_collected": self.samples_collected,
                "x_axis/rollouts": self.i + 1,
                # Training metrics
                "Training/Loss": loss_mean,
                "Training/Accuracy": mean_accuracy, 
                "Training/learning_rate": optimizer.param_groups[0]['lr'],
                "Training/log_std_mean": log_sigma.mean().item(),
                "Training_collected/ep_reward": self.collected_ep_rewards if hasattr(self, 'collected_ep_rewards') else None,
                "Training_collected/accuracy": self.collected_accuracy if hasattr(self, 'collected_accuracy') else None,
                # Stats
                "Stats/epoch_time_seconds": epoch_time,
                "Stats/samples_per_second": samples_per_sec,
                # Samples seen
                "Training/new_samples_seen": self.new_samples,
                "Training/old_samples_seen": self.old_samples,
                "Training/loss_old": self.loss_old,
                "Training/loss_new": self.loss_new,
            } | buffer_metrics

            # Test based on optimization steps (always relative to the absolute target)
            if (len(test_optimized_step) > 0 and self.global_step >= test_optimized_step[0])  or get_total_optimization_steps() >= optimization_steps_to_do:
                test_optimized_step = [x for x in test_optimized_step if x >  self.global_step]
                print(f"\nRunning evaluation at step {get_total_optimization_steps():,} (rollout {self.i+1})...")

                eval_start_time = time.time()
                mean_reward_ID, accuracy_ID, mean_reward_OD, accuracy_OD, infos = self.test()
                eval_time = time.time() - eval_start_time
                print(f"Evaluation completed in {eval_time:.1f}s - ID: {mean_reward_ID:.2f}, OD: {mean_reward_OD:.2f}")

                infos_to_log = infos_to_log | infos

                # Generalization gaps
                infos_to_log["Generalization/ID_OD_reward_gap"] = mean_reward_ID - mean_reward_OD
                infos_to_log["Generalization/ID_OD_accuracy_gap"] = accuracy_ID - accuracy_OD

                if mean_reward_ID > best_ID_reward:
                    best_ID_reward = mean_reward_ID
                    print(f"New best ID reward: {best_ID_reward:.2f} at step {get_total_optimization_steps():,}")
                    self.save_model(mean_ID_reward=mean_reward_ID, seed_ID=self.dict_test_enviroment['IDseeds'], mean_reward_OD=mean_reward_OD, seed_OD=self.dict_test_enviroment['ODseeds'], path=self.second_phase_folder, title=f"{self.loss_type}_BestModel_best_ID", wandb_bool=True)

                self.save_model(mean_ID_reward=mean_reward_ID, seed_ID=self.dict_test_enviroment['IDseeds'], mean_reward_OD=mean_reward_OD, seed_OD=self.dict_test_enviroment['ODseeds'], path=self.second_phase_folder, title=f"{self.loss_type}_step_{get_total_optimization_steps()}", wandb_bool=True)
                self.Student.train()

            self._load_infos(infos_to_log)
            self.i += 1

        # Log final summary
        wandb.log({
            "Training/final_best_ID_reward": best_ID_reward,
            "Params/total_optimization_steps": get_total_optimization_steps(),
            "Params/total_rollouts": self.i,
        })
        wandb.finish()

    def _update_step(self, dataloader, optimizer, max_steps=None):
        """Run one epoch of training or up to max_steps batch updates.
        
        Args:
            max_steps: Maximum number of batches to process. If None, process the entire dataloader.
        """
        Student_keys = self.Student.input_type
        epoch_start_time = time.time()
        loss_epoch = 0
        total_accuracy = 0
        samples_processed = 0
        batches_processed = 0

        self.loss_old = 0.0
        self.loss_new = 0.0
        self.old_samples = 0
        self.new_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            # Stop if we reached max_steps
            if max_steps is not None and batch_idx >= max_steps:
                break

            if batch_idx % 100 == 0:
                if batch_idx == 0: 
                    print(f"Epoch {self.i+1}, processing batch {batch_idx+1}/{len(dataloader)} | Global step: {self.global_step:,}", end='\r')
                else:
                    print(f"Epoch {self.i+1}, processing batch {batch_idx+1}/{len(dataloader)} - Time for last 100 batches: {time.time() - batch_start_time:.2f}s | Global step: {self.global_step:,}", end='\r')
                batch_start_time = time.time()
            
            obs_batch, actions_T_batch, logits_distribution_T_batch, weights, idxs, masks_old, maks_new = batch

            if len(Student_keys) == 1:
                input_S = obs_batch[Student_keys[0]].to(self.device, non_blocking=True)
            else:
                input_S = tuple([obs_batch[k].to(self.device, non_blocking=True) for k in Student_keys])
            action_S, _, _ = self.Student.get_action(input_S)

            if self.alpha > 0.0:
                loss = self.loss_fn.compute_loss(action_S, actions_T_batch, self.Student, self.Teacher, logits_distribution_T_batch, reduction = 'none')
                self.Buffer.update_errors(idxs, loss.detach().cpu())

                loss_tensor = loss * weights.to(self.device, non_blocking=True)
                loss = loss_tensor.mean()
                self.loss_old +=loss_tensor[masks_old].mean() if masks_old.sum() > 0 else torch.tensor(0.0, device=self.device)
                self.loss_new += loss_tensor[maks_new].mean() if maks_new.sum() > 0 else torch.tensor(0.0, device=self.device)
                self.old_samples += masks_old.sum().item()
                self.new_samples += maks_new.sum().item()
            else:
                loss = self.loss_fn.compute_loss(action_S, actions_T_batch, self.Student, self.Teacher, logits_distribution_T_batch, reduction = 'none')
                loss = (loss * weights.to(self.device, non_blocking=True)).mean()
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Student.parameters(), max_norm=1.0)
            optimizer.step()

            samples_processed += actions_T_batch.size(0)
            self.global_step += 1  # Counts batches (optimization steps), not samples
            batches_processed += 1
            total_accuracy += get_accuracy(action_S, actions_T_batch.to(self.device, non_blocking=True), self.envs)
            loss_epoch += loss.detach()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()  

        epoch_time = time.time() - epoch_start_time
        loss_mean = (loss_epoch / batches_processed) if batches_processed > 0 else 0
        self.loss_old = (self.loss_old / batches_processed) if self.old_samples > 0 else 0
        self.loss_new = (self.loss_new / batches_processed) if self.new_samples > 0 else 0
        mean_accuracy = (total_accuracy / batches_processed) if batches_processed > 0 else 0
        samples_per_sec = samples_processed / epoch_time if epoch_time > 0 else 0
        return loss_mean, mean_accuracy, epoch_time, samples_per_sec, samples_processed

    @torch.no_grad()
    def _collect_rollouts(self, steps):
        self.Teacher.eval()
        self.Teacher.deterministic = True
        self.Student.eval()
        self.Student.deterministic = True
        Ep_rewards = 0
        finished_episodes = 0
        total_accuracy = 0
        total_actions = 0

        beta = max(1 - 1 * self.global_step / (0.3*self.step_to_be_done_in_phase_two) , 0)
        beta = max( 0.95**(self.i) , 0.15)
        beta = 0
        who_take_actoins = (torch.rand(steps, self.envs.num_envs) < beta).long()
        log_std_T_mean, log_std_S_mean = torch.zeros(steps), torch.zeros(steps)
        entropy_T_mean, entropy_S_mean = torch.zeros(steps), torch.zeros(steps)
        teachers_actions = 0
        students_actions = 0

        for step in range(steps):

            input_S = get_input(self.Student, self.next_done, self.info)
            action_S, logprob_S, entropy_S = self.Student.get_action(input_S)

            input_T = get_input(self.Teacher, self.next_done, self.info)
            action_T, logprob_T, entropy_T = self.Teacher.get_action(input_T)

            if self.Teacher.action_type == 'Continuous':
                logits = self.Teacher.get_logits()
                _, log_std = logits.chunk(2, dim=-1)
                log_std_T_mean[step] = log_std.mean()

                logits = self.Student.get_logits()
                _, log_std_S = logits.chunk(2, dim=-1)
                log_std_S_mean[step] = log_std_S.mean()
            else:
                entropy_T_mean[step] = entropy_S.mean()
                entropy_S_mean[step] = entropy_T.mean()

            total_accuracy += get_accuracy(action_S, action_T.to(self.device, non_blocking=True), self.envs)
            total_actions += action_S.size(0)

            self.b_actions_T[step * self.envs.num_envs : (step+1) * self.envs.num_envs] = action_T
            self.b_logits_distribution_T[step * self.envs.num_envs : (step+1) * self.envs.num_envs] = self.Teacher.get_logits()
            loss = self.loss_fn.compute_loss(action_S, action_T, self.Student, self.Teacher, reduction='none')
            for k in self.b_new_observations.keys():
                if k != 'Frame':
                    self.b_new_observations[k][step * self.envs.num_envs : (step+1) * self.envs.num_envs] = torch.from_numpy(self.info[k])
                if k == 'Frame':
                    if 'Frame' in self.Student.input_type:
                        self.b_new_observations['Frame'][step * self.envs.num_envs : (step+1) * self.envs.num_envs] = self.Student.input_manager.get_input()
                    elif 'Frame' in self.Teacher.input_type:
                        self.b_new_observations['Frame'][step * self.envs.num_envs : (step+1) * self.envs.num_envs] = self.Teacher.input_manager.get_input()
                    else:
                        raise ValueError("Frame observations not found in either Student or Teacher input types.")
                    
            
            self.b_loss[step * self.envs.num_envs : (step+1) * self.envs.num_envs] = loss.detach() 

            # Environment step
            condition = who_take_actoins[step].bool()
            # condition =log_std_S.mean(axis=1) > -2.0
            if action_S.dim() != 1:
                condition = condition.unsqueeze(-1)
            actions_for_step = torch.where(condition.cpu(), action_T.cpu(), action_S.cpu())
            students_actions += (~condition.cpu()).sum().item()
            teachers_actions += (condition.cpu()).sum().item()
            
            next_obs, reward, terminations, truncations, self.info = self.envs.step(actions_for_step.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.next_done = torch.tensor(next_done, device=self.device)

            if next_done.sum() >0:
                Ep_rewards += self.info['episode']['r'].sum() 
                finished_episodes += next_done.sum()  

        self.collected_ep_rewards = Ep_rewards / finished_episodes if finished_episodes > 0 else None
        self.collected_accuracy = total_accuracy / total_actions if total_actions > 0 else 0

        for step in range(steps):
            wandb.log({
                'x_axis/epochs': self.i+1 + step/(steps+1),
                'Training_collected/log_std_teacher' : log_std_T_mean[step],
                'Training_collected/log_std_student': log_std_S_mean[step],
                'Training_collected/beta_rollout' : beta
            })

        if self.Teacher.action_type == 'Continuous': 
            wandb.log({
                'x_axis/epochs': self.i+1,
                'Training_collected/teachers_actions': teachers_actions,
                'Training_collected/students_actions': students_actions,
            })
        else:
            wandb.log({
                'x_axis/epochs': self.i+1,
                'Training_collected/entropy_teacher' : entropy_T_mean.mean().item(),
                'Training_collected/entropy_student': entropy_S_mean.mean().item(),
            })

    def test(self):
        if "Frame" in self.Student.input_type:
            stored_input_S = self.Student.get_input_manager()
            self.Student.reset_input_manager()
        if "Frame" in self.Teacher.input_type:
            stored_input_T = self.Teacher.get_input_manager()
            self.Teacher.reset_input_manager()
        
        self.test_envs = self.vec_env_fun([make_env(self.dict_test_enviroment, idx, wrappers=self.wrapper) for idx in range(len(self.dict_test_enviroment['IDseeds']))], autoreset_mode = 'Disabled')
        mean_reward_ID, info_ID_Student = test_model(model_A=self.Student, model_B = self.Teacher, env =self.test_envs, iteration=self.i, global_step = self.global_step, seeds=self.dict_test_enviroment['IDseeds'], video_folder=self.dict_test_enviroment['video_folder'], title = 'ID test', loss_fn = self.loss_fn.compute_loss, wandb_bool=False)
        accuracy_ID = self.Tester_under_teacher_ID.TestModel(self.Student)
        
        if "Frame" in self.Student.input_type:
            self.Student.reset_input_manager()
        if "Frame" in self.Teacher.input_type:
            self.Teacher.reset_input_manager()

        self.test_envs = self.vec_env_fun([make_env(self.dict_test_enviroment, idx, wrappers=self.wrapper) for idx in range(len(self.dict_test_enviroment['IDseeds']))], autoreset_mode = 'Disabled')
        mean_reward_OD, info_OD_Student = test_model(model_A=self.Student, model_B = self.Teacher, env =self.test_envs, iteration=self.i, global_step = self.global_step, seeds=self.dict_test_enviroment['ODseeds'], video_folder=self.dict_test_enviroment['video_folder'], title = 'OD test', loss_fn = self.loss_fn.compute_loss, wandb_bool=False)
        accuracy_OD = self.Tester_under_teacher_OD.TestModel(self.Student)

        if "Frame" in self.Student.input_type:
            self.Student.set_input_manager(stored_input_S)
        if "Frame" in self.Teacher.input_type:
            self.Teacher.set_input_manager(stored_input_T)

        infos = info_ID_Student | info_OD_Student | {
            'Test/Accuracy under Teacher ID': accuracy_ID,
            'Test/Accuracy under Teacher OD': accuracy_OD,
        }

        return mean_reward_ID, accuracy_ID, mean_reward_OD, accuracy_OD, infos
    
    def save_model(self, mean_ID_reward, seed_ID, mean_reward_OD, seed_OD, path, title, wandb_bool=True):
        info = {
            'mean_ID_reward': mean_ID_reward,
            'seed_ID': seed_ID,
            'mean_OD_reward': mean_reward_OD,
            'seed_OD': seed_OD,
            "optimization_steps": self.global_step,
            "samples_seen": self.samples_seen,
            "epochs": self.i + 1,
            'total_optimization_steps': self.optimization_steps if hasattr(self, 'optimization_steps') else None,
        }
        self.df_info.to_csv(os.path.join(path, f"{self.run_name}_metrics.csv"), index=False)
        self.Student.save_model(path=path, title=title, wandb_bool=wandb_bool, info_dict=info)

    def _load_infos(self, infos):
        infos['epoch'] = self.i + 1
        infos['global_step'] = self.global_step

        def to_scalar(v):
            if isinstance(v, torch.Tensor):
                return v.item()
            if isinstance(v, np.generic):
                return v.item()
            return v

        scalar_infos = {k: to_scalar(v) for k, v in infos.items() 
                        if not isinstance(v, (wandb.Table, wandb.Image))}
        self.df_info = pd.concat([self.df_info, pd.DataFrame([scalar_infos])], ignore_index=True)
        

        if self.alpha > 0.0 and self.i % 1 == 0 and hasattr(self.Buffer, 'priorities'): 
            prob = self.Buffer.priorities.cpu().numpy() / self.Buffer.priorities.sum().item()
            epoch_df = pd.DataFrame({
                'epoch': [self.i + 1] * len(prob),
                'dataset_index': range(len(prob)),
                'probability': prob
            })
            table = wandb.Table(dataframe=epoch_df)
            infos[f'Table/Error_Distribution_table_epoch_{self.i+1}'] = table

            # Scatter plot: dataset_index vs probability
            fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
            ax_scatter.scatter(epoch_df['dataset_index'], epoch_df['probability'], alpha=0.6, s=10)
            ax_scatter.set_xlabel('Dataset Index')
            ax_scatter.set_ylabel('Probability')
            ax_scatter.set_title(f"Scatter Plot Step {self.global_step} (Alpha={self.alpha})")
            ax_scatter.grid(True, alpha=0.3)
            infos['Training/Error_Distribution_scatter'] = wandb.Image(fig_scatter)

            # Histogram: probability distribution
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            ax_hist.hist(prob, bins=50, edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel('Probability')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title(f"Probability Histogram Step {self.global_step} (Alpha={self.alpha})")
            ax_hist.grid(True, alpha=0.3)
            infos['Training/Error_Distribution_histogram'] = wandb.Image(fig_hist)
            
            plt.close('all')

        wandb.log(infos)
 
    



