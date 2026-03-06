import torch
import numpy as np
import gymnasium as gym
import wandb
import time
import os
import pandas as pd
from Algorithm.Test_model import test_model, TestModel_underTeacher
from Enviroment.Utils import make_env
from Algorithm.Utils import get_input, set_seeds, create_observation_dict, load_to_observatoin_dict, get_accuracy
from Algorithm.seeds import get_training_seed



class Student_Distillation:
    def __init__(
        self,
        student,
        teacher,
        path_folder,
        dict_enviroment,
        device,
        config,
        dict_test_enviroment = None,
        run_index : int = 0
    ):
        self.path_folder = path_folder
        self.run_name = dict_enviroment["run_name"] + f"_run{run_index}"
        self.env_name = dict_enviroment["env_name"]
        self.device = device
        self.config = config

        # --- Setup envs ---
        wrapper = dict_enviroment.get('wrappers', [])
        if self.env_name  == "Pusher-v5":
            self.vector_env_fn = gym.vector.SyncVectorEnv
        else:
            self.vector_env_fn = gym.vector.AsyncVectorEnv
        self.envs = self.vector_env_fn([make_env(dict_enviroment, idx, wrappers=wrapper) for idx in range(config["num_envs"])])
        if dict_test_enviroment is not None:
            self.dict_test_enviroment = dict_test_enviroment
            test_envs = self.vector_env_fn([make_env(self.dict_test_enviroment, idx, wrappers=self.dict_test_enviroment.get('wrappers', [])) for idx in range(len(self.dict_test_enviroment['ODseeds']))], autoreset_mode = 'Disabled')
            self.Tester_under_teacher_OD = TestModel_underTeacher(Teacher = teacher, Student=student, seeds = dict_test_enviroment['ODseeds'], envs = test_envs)
            if 'Frame' in student.input_type:
                student.input_manager = None

        # --- Select agent type ---
        self.Student = student.to(device)
        self.optimizer = torch.optim.Adam(self.Student.parameters(), lr=config["learning_rate"], eps=1e-5)
        self.Teacher = teacher.to(device)

        # --- Initialize buffers ---
        self.obs_Teacher = create_observation_dict((config["num_steps"], config["num_envs"]), model = self.Teacher, device=device)
        self.obs_Student = create_observation_dict((config["num_steps"], config["num_envs"]), model = self.Student, device=device)
        self.key_student = self.Student.input_type
        self.key_teacher = self.Teacher.input_type

        action_shape = self.envs.single_action_space.shape
        self.actions = torch.zeros((config["num_steps"], config["num_envs"]) + action_shape, device=device)
        self.actions_other = torch.zeros((config["num_steps"], config["num_envs"]) + action_shape, device=device)
        self.logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=device)

        # --- Environment state ---
        _, self.info = self.envs.reset(seed=get_training_seed(run_index))
        set_seeds(get_training_seed(run_index))
        self.next_obs_S = { k: torch.tensor(self.info[k], device=device) for k in self.key_student}
        self.next_obs_T = { k: torch.tensor(self.info[k], device=device) for k in self.key_teacher}
        self.next_done = torch.zeros(config["num_envs"], device=device)

        self.global_step = 0
        self.optimization_steps = 0  # number of gradient updates
        self.samples_seen = 0  # samples used for training (minibatch_size × updates)
        self.samples_collected = 0  # samples collected from rollouts
        self.start_time = time.time()
        
        # Create folder for this run
        self.run_folder = os.path.join(path_folder, self.run_name)
        os.makedirs(self.run_folder, exist_ok=True)
        
        # DataFrame to store all metrics
        self.df_metrics = pd.DataFrame()

        config_log = {**self.config, **dict_enviroment}
        if dict_test_enviroment is not None:
            config_log.update(dict_test_enviroment)
        wandb.init(
            entity='Distillation_RL',
            project=f"{self.env_name}",
            name=f"{self.run_name}",
            config=config_log,
            dir=self.path_folder,
            save_code=True,
        )
        self.mode = 'student'
        self.info_to_log = {}

    def _set_model_modes(self):
        """Set standard model modes: Teacher to eval/deterministic, Student to train/non-deterministic."""
        self.Teacher.eval()
        self.Teacher.deterministic = True
        self.Student.train()
        self.Student.deterministic = False
    
    def _format_model_input(self, obs_dict, keys, indices=None):
        """Format model input from observation dict (single or tuple)."""
        if len(keys) == 1:
            return obs_dict[keys[0]] if indices is None else obs_dict[keys[0]][indices]
        else:
            if indices is None:
                return tuple([obs_dict[k] for k in keys])
            else:
                return tuple([obs_dict[k][indices] for k in keys])
    
    @torch.inference_mode()
    def collect_rollouts(self, mode):
        cfg = self.config
        self._set_model_modes()
        reward_ended_episodes = []

        for step in range(cfg["num_steps"]):
            # Prepare input
            input_S = get_input(self.Student, self.next_done, self.info)
            load_to_observatoin_dict(self.obs_Student, input_S, step)

            input_T = get_input(self.Teacher, self.next_done, self.info)
            load_to_observatoin_dict(self.obs_Teacher, input_T, step)

            # Action
            if mode == 'student':
                action, logprob, _ = self.Student.get_action(input_S)
                self.actions_other[step] = self.Teacher.get_action(input_T)[0].detach()
            elif mode == 'teacher':
                action, logprob, _ = self.Teacher.get_action(input_T)
                self.actions_other[step] = self.Student.get_action(input_S)[0].detach()

            self.actions[step] = action
            self.logprobs[step] = logprob

            # Environment step
            _, reward, terminations, truncations, self.info = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.next_done = torch.tensor(next_done, device=self.device)
            self.next_obs_S = { k: torch.tensor(self.info[k], device=self.device) for k in self.key_student}
            self.next_obs_T = { k: torch.tensor(self.info[k], device=self.device) for k in self.key_teacher}

            if next_done.sum() > 0:
                reward_ended_episodes.extend(self.info['episode']['r'][next_done])

        self.info_to_log['training/avg_reward'] = float(np.mean(reward_ended_episodes)) if reward_ended_episodes else float('nan')
        self.info_to_log['training/avg_accuracy'] = get_accuracy(self.actions, self.actions_other, self.envs)
        self.info_to_log['iteration'] = self.iteration
        
        # Update samples_collected (samples from rollouts)
        self.samples_collected += cfg["num_steps"] * cfg["num_envs"]
        
        return reward_ended_episodes
    
    def collect_rollouts_aux(self):
        return self.collect_rollouts('student')

    def log_and_reset(self):
        """Log all accumulated info to wandb and reset the dictionary."""
        # X-axis metrics (use any of these in wandb)
        self.info_to_log['x_axis/optimization_steps'] = self.optimization_steps
        self.info_to_log['x_axis/samples_seen'] = self.samples_seen
        self.info_to_log['x_axis/samples_collected'] = self.samples_collected
        self.info_to_log['x_axis/epochs'] = self.iteration
        # Stats
        self.info_to_log['training/SPS'] = int(self.global_step / (time.time() - self.start_time))
        self.info_to_log['training/learning_rate'] = self.optimizer.param_groups[0]["lr"]
        self.info_to_log['global_step'] = self.global_step
        # Test
        self.info_to_log = self.info_to_log | self.info_test_OD
        
        # Save to DataFrame (only scalar values)
        scalar_metrics = {k: v for k, v in self.info_to_log.items() if isinstance(v, (int, float, np.number))}
        self.df_metrics = pd.concat([self.df_metrics, pd.DataFrame([scalar_metrics])], ignore_index=True)
        
        wandb.log(self.info_to_log, step=self.global_step)
        self.info_to_log = {}

    def update(self):
        cfg = self.config
        self._set_model_modes()

        # Flatten
        b_obs_Student = { k: self.obs_Student[k].reshape((-1,) + self.obs_Student[k].shape[2:]) for k in self.key_student}
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)

        b_obs_Teacher = { k: self.obs_Teacher[k].reshape((-1,) + self.obs_Teacher[k].shape[2:]) for k in self.key_teacher}
        with torch.no_grad():
            b_input_Teacher = self._format_model_input(b_obs_Teacher, self.key_teacher)
            _, _, _ = self.Teacher.get_action(b_input_Teacher)  # To update logits
        b_logits_teacher = self.Teacher.get_logits()

        num_samples = b_actions.shape[0]
        b_inds = np.arange(num_samples)

        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                b_input_Student = self._format_model_input(b_obs_Student, self.key_student, mb_inds)
                _, newlogprob, entropy = self.Student.get_action(b_input_Student, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                if start == 0 and epoch == 0 and self.mode == 'student':
                    if not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).all(): 
                        print(f"Warning: Ratio not close to 1 at first update! {(not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).sum().item())} elements differ. {torch.abs(ratio - 1.0).max().item()} max diff.")

               
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                ###################### Distillation Loss
                Teacher_distribution = self.Teacher.create_distribution_from_logits(b_logits_teacher[mb_inds])
                Student_distribution = self.Student.get_last_distribution()
                Distillation_loss = torch.distributions.kl.kl_divergence(Teacher_distribution, Student_distribution).mean()
                ########################

                entropy_loss = entropy.mean()
                anneal_factor = cfg.get("anneal_factor", 0.99)
                curr_ent_coef = cfg["ent_coef"] * (anneal_factor ** self.iteration if cfg["anneal_ent_coef"] else 1)
                loss = Distillation_loss - curr_ent_coef * entropy_loss 

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.Student.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                
                # Update counters
                self.optimization_steps += 1
                self.samples_seen += len(mb_inds)
        
        self.info_to_log.update({
            'training/entropy' : entropy_loss.item(),
            'training/approx_kl' : approx_kl.item(),
            'training/grad_norm' : grad_norm.item(),
            'training/ent_coef' : curr_ent_coef,
            'training/Distillation_loss' : Distillation_loss.item(),
        })

    def train(self):
        cfg = self.config
        start_time = time.time()
        print('Start training...')
        optimization_steps = 0
        max_optimization_steps = cfg.get("optimization_steps", cfg["num_iterations"] * cfg["update_epochs"])  # fallback for backward compatibility
        self.iteration = 1
        while optimization_steps < max_optimization_steps:
            if cfg["anneal_lr"]:
                frac = 1.0 - (optimization_steps) / max_optimization_steps
                self.optimizer.param_groups[0]["lr"] = frac * cfg["learning_rate"]

            reward_episodes = self.collect_rollouts_aux()
            self.global_step += cfg["num_envs"] * cfg["num_steps"]
            steps_this_iter = self.update_with_count()
            optimization_steps += steps_this_iter

            if self.iteration % self.config['Verbose_frequency'] == 0:
                print(f"[OptStep {optimization_steps}/{max_optimization_steps}] AvgRew: {self.info_to_log['training/avg_reward']:.2f}")

            if self.iteration % self.config["Test_frequency"] == 0 or self.iteration == 1 or optimization_steps >= max_optimization_steps:
                mean_rwd_test_OD = self.test()
                info = {
                    "training/avg_reward": self.info_to_log['training/avg_reward'],
                    "test/avg_reward_OD": mean_rwd_test_OD,
                    "optimization_steps": optimization_steps,
                }
                self._save_metrics_df()
                if self.iteration != 1: 
                    self.Student.save_model(path=self.run_folder, title=f"checkpoint_opt{optimization_steps}", wandb_bool=True, info_dict=info)
                self.Student.train()
                self.Student.deterministic = False

            self.log_and_reset()
            self.iteration += 1

        self._save_metrics_df()
        print("Training completed.", time.time() - start_time)
        self.envs.close()

    def update_with_count(self):
        cfg = self.config
        self._set_model_modes()
        b_obs_Student = { k: self.obs_Student[k].reshape((-1,) + self.obs_Student[k].shape[2:]) for k in self.key_student}
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_obs_Teacher = { k: self.obs_Teacher[k].reshape((-1,) + self.obs_Teacher[k].shape[2:]) for k in self.key_teacher}
        with torch.no_grad():
            b_input_Teacher = self._format_model_input(b_obs_Teacher, self.key_teacher)
            _, _, _ = self.Teacher.get_action(b_input_Teacher)  # To update logits
        b_logits_teacher = self.Teacher.get_logits()
        num_samples = b_actions.shape[0]
        b_inds = np.arange(num_samples)
        minibatch_updates = 0
        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_inds = b_inds[start:end]
                b_input_Student = self._format_model_input(b_obs_Student, self.key_student, mb_inds)
                _, newlogprob, entropy = self.Student.get_action(b_input_Student, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                Teacher_distribution = self.Teacher.create_distribution_from_logits(b_logits_teacher[mb_inds])
                Student_distribution = self.Student.get_last_distribution()
                Distillation_loss = torch.distributions.kl.kl_divergence(Teacher_distribution, Student_distribution).mean()
                entropy_loss = entropy.mean()
                anneal_factor = cfg.get("anneal_factor", 0.99)
                curr_ent_coef = cfg["ent_coef"] * (anneal_factor ** getattr(self, 'iteration', 1) if cfg["anneal_ent_coef"] else 1)
                loss = Distillation_loss - curr_ent_coef * entropy_loss 
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.Student.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                minibatch_updates += 1
        self.optimization_steps += minibatch_updates
        self.samples_seen += minibatch_updates * cfg["minibatch_size"]
        self.info_to_log.update({
            'training/entropy' : entropy_loss.item(),
            'training/approx_kl' : approx_kl.item(),
            'training/grad_norm' : grad_norm.item(),
            'training/ent_coef' : curr_ent_coef,
            'training/Distillation_loss' : Distillation_loss.item(),
        })
        return minibatch_updates

    def _save_metrics_df(self):
        """Save the metrics DataFrame to a CSV file."""
        csv_path = os.path.join(self.run_folder, "metrics.csv")
        self.df_metrics.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

    def test(self):
        if "Frame" in self.Student.input_type:
            stored_input_S = self.Student.get_input_manager()
        if "Frame" in self.Teacher.input_type:
            stored_input_T = self.Teacher.get_input_manager()

        self.test_envs = self.vector_env_fn([make_env(self.dict_test_enviroment, idx, wrappers=self.dict_test_enviroment.get('wrappers', [])) for idx in range(len(self.dict_test_enviroment['ODseeds']))], autoreset_mode = 'Disabled')
        mean_reward_OD, self.info_test_OD = test_model(model_A=self.Student, model_B = self.Teacher, env =self.test_envs, iteration=self.iteration, global_step = self.global_step, seeds=self.dict_test_enviroment['ODseeds'], video_folder=self.dict_test_enviroment['video_folder'], title = 'OD test')
        accuracy_OD = self.Tester_under_teacher_OD.TestModel(self.Student)

        if "Frame" in self.Student.input_type:
            self.Student.set_input_manager(stored_input_S)
        if "Frame" in self.Teacher.input_type:
            self.Teacher.set_input_manager(stored_input_T)

        self.info_to_log['test/accuracy_OD_under_T'] = accuracy_OD
        return mean_reward_OD
    
class Teacher_Distillation(Student_Distillation):
    def __init__(
        self,
        student,
        teacher,
        path_folder,
        dict_enviroment,
        device,
        config,
        dict_test_enviroment = None,
        run_index : int = 0
    ):
        super().__init__(
            student,
            teacher,
            path_folder,
            dict_enviroment,
            device,
            config,
            dict_test_enviroment,
            run_index=run_index
        )
        self.Teacher = teacher.to(device)
        self.mode = 'teacher'

    def collect_rollouts_aux(self):
        return self.collect_rollouts('teacher')