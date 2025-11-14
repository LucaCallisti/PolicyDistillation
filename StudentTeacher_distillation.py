import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import wandb
import time
from Utils_PPO import set_seeds
from Test_model import test_model, TestModel_underTeacher
import time
from PPO import make_env
from My_wrapper import RenderFrameWrapper



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
        project_prefix="StudentDistillation",
    ):
        self.path_folder = path_folder
        self.run_name = dict_enviroment["run_name"]
        self.env_name = dict_enviroment["env_name"]
        self.device = device
        self.config = config

        # --- Setup envs ---
        self.envs = gym.vector.AsyncVectorEnv([make_env(dict_enviroment, idx, wrappers=[RenderFrameWrapper]) for idx in range(config["num_envs"])])
        if dict_test_enviroment is not None:
            self.test_envs = gym.vector.AsyncVectorEnv([make_env(dict_test_enviroment, idx, wrappers=[RenderFrameWrapper]) for idx in range(len(dict_test_enviroment['IDseeds']))], autoreset_mode = 'Disabled')
            self.dict_test_enviroment = dict_test_enviroment
            self.Tester_under_teacher_ID = TestModel_underTeacher(Teacher = teacher, seeds = dict_test_enviroment['IDseeds'], envs = self.test_envs)
            self.Tester_under_teacher_OD = TestModel_underTeacher(Teacher = teacher, seeds = dict_test_enviroment['ODseeds'], envs = self.test_envs)


        # --- Select agent type ---
        self.student = student.to(device)
        self.optimizer = optim.Adam(self.student.parameters(), lr=config["learning_rate"], eps=1e-5)
        self.Teacher = teacher.to(device)

        # --- Initialize buffers ---
        obs_shape = self.student.input_shape
        action_shape = self.envs.single_action_space.shape

        self.obs = torch.zeros((config["num_steps"], config["num_envs"]) + obs_shape, device=device)
        self.obs_Teacher = torch.zeros((config["num_steps"], config["num_envs"]) + self.Teacher.input_shape, device=self.device)
        self.actions = torch.zeros((config["num_steps"], config["num_envs"]) + action_shape, device=device)
        self.logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=device)

        # --- Environment state ---
        self.next_obs, self.info = self.envs.reset(seed=config["seed"])
        set_seeds(config['seed'][0]) if isinstance(config['seed'], list) else set_seeds(config['seed'])
        self.next_obs = torch.tensor(self.next_obs, device=device)
        self.next_done = torch.zeros(config["num_envs"], device=device)

        self.global_step = 0
        self.start_time = time.time()

        config_log = {**self.config, **dict_enviroment}
        if dict_test_enviroment is not None:
            config_log.update(dict_test_enviroment)
        wandb.init(
            project=f"{project_prefix}_{self.env_name}",
            name=self.run_name,
            config=config_log,
            dir=self.path_folder,
            save_code=True,
        )
 
    @torch.inference_mode()
    def collect_rollouts(self, agent):
        cfg = self.config
        reward_ended_episodes = []

        for step in range(cfg["num_steps"]):
            # Prepara input
            if agent.agent_type == "Image":
                screen = agent.get_screen(envs=None, screen=self.next_obs.cpu())
                input_ = agent.update_memory(screen, self.next_done.cpu()).to(self.device)
            else:
                input_ = self.next_obs.to(self.device)

            self.obs[step] = input_

            # Azione
            action, logprob, _, value = agent.get_action_and_value(input_)
            self.actions[step] = action
            self.logprobs[step] = logprob

            # Step ambiente
            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            self.next_obs, self.next_done = torch.tensor(next_obs, device=self.device), torch.tensor(next_done, device=self.device)

            if next_done.sum() > 0:
                reward_ended_episodes.extend(infos['episode']['r'][next_done])

        avg_rew = float(np.mean(reward_ended_episodes)) if reward_ended_episodes else float('nan')
        return avg_rew, reward_ended_episodes
    
    def collect_rollouts_aux(self):
        return self.collect_rollouts(self.student)

    def update(self):
        cfg = self.config

        # Flatten
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)

        b_obs_Teacher = self.obs_Teacher.reshape((-1,) + self.obs_Teacher.shape[2:])
        with torch.no_grad():
            _, _, _, b_Teacher_values = self.Teacher.get_action_and_value(b_obs_Teacher)
        b_logits_teacher = self.Teacher.get_logits()

        num_samples = b_obs.shape[0]
        b_inds = np.arange(num_samples)

        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.student.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Sanity check
                if start == 0 and epoch == 0:
                    if not torch.isclose(ratio, torch.tensor(1.0), atol=1e-2).all(): 
                        print(f"Warning: Ratio not close to 1 at first update! elements differ. {torch.abs(ratio - 1.0).max().item()} max diff.")
                    
                # Only for metric logging
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                ###################### Distillation Loss
                Teacher_distribution = self.Teacher.create_distribution_from_logits(b_logits_teacher[mb_inds])
                Student_distribution = self.student.get_last_distribution()
                Distillation_loss = torch.distributions.kl.kl_divergence(Teacher_distribution, Student_distribution).mean()
                ########################

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_Teacher_values[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                curr_ent_coef = cfg["ent_coef"] * (0.99 ** self.iteration if cfg["anneal_ent_coef"] else 1)
                loss = - Distillation_loss- curr_ent_coef * entropy_loss + cfg["vf_coef"] * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.student.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

        return {
            'training/v_loss' : v_loss.item(),
            'training/entropy' : entropy_loss.item(),
            'training/approx_kl' : approx_kl.item(),
            'training/grad_norm' : grad_norm.item(),
            'training/ent_coef' : curr_ent_coef,
            'training/Distillation_loss' : Distillation_loss.item(),
        }

    def train(self):
        cfg = self.config
        n_better = 0

        start_time = time.time()
        print('Start training...')
        for self.iteration in range(1, cfg["num_iterations"] + 1):
            if cfg["anneal_lr"]:
                frac = 1.0 - (self.iteration - 1.0) / cfg["num_iterations"]
                self.optimizer.param_groups[0]["lr"] = frac * cfg["learning_rate"]

            avg_rew, reward_episodes = self.collect_rollouts_aux()
            self.global_step += cfg["num_envs"] * cfg["num_steps"]
            logs = self.update()

            rewards_np = np.array(reward_episodes)
            if (rewards_np > cfg["target_reward"]).all():
                n_better += len(reward_episodes)
            else:
                mask = rewards_np > cfg["target_reward"]
                false_idx = np.where(mask == False)[0]
                n_better = rewards_np.size - false_idx[-1] - 1

            if n_better >= 100:
                print(f"Solved at iteration {self.iteration}: Avg reward = {avg_rew:.2f}")
                mean_rwd_test_ID, mean_rwd_test_OD = self.test()
                info = {
                    "training/avg_reward": avg_rew,
                    "test/avg_reward_ID": mean_rwd_test_ID,
                    "test/avg_reward_OD": mean_rwd_test_OD,
                    "iteration": self.iteration,
                }
                self.student.save_model(path = self.path_folder, title = self.run_name + "_ppo", wandb_bool = True, info_dict = info)
                break
            
            if self.iteration % self.config['Verbose_frequency'] == 0:
                print(f"[{self.iteration}/{cfg['num_iterations']}] AvgRew: {avg_rew:.2f}")
            
            if self.iteration % self.config["Test_frequency"] == 0 or self.iteration == 1:
                mean_rwd_test_ID, mean_rwd_test_OD = self.test()
                info = {
                    "training/avg_reward": avg_rew,
                    "test/avg_reward_ID": mean_rwd_test_ID,
                    "test/avg_reward_OD": mean_rwd_test_OD,
                    "iteration": self.iteration,
                }
                if self.iteration != 1: self.student.save_model(path = self.path_folder, title = self.run_name + f"it{self.iteration}_ppo", wandb_bool = True, info_dict = info)
                self.student.train()
                self.student.deterministic = False

                if self.config["reset_after_test"]:
                    next_obs, _ = self.envs.reset()
                    self.next_obs = torch.tensor(next_obs, device=self.device)
                    init_screen = self.student.get_screen(screen = self.next_obs)
                    _ = self.student.reset_memory(init_screen)

            # Logging
            wandb.log({
                "training/avg_reward": avg_rew,
                **logs,
                "training/SPS": int(self.global_step / (time.time() - self.start_time)),
                'training/n_better_episodes': n_better,
                "iteration": self.iteration,
                'global_step': self.global_step
            }, step=self.global_step)


        print("Training completed.", time.time() - start_time)
        self.envs.close()


    def test(self):
        mean_reward_ID = test_model(model_A=self.student, model_B = self.Teacher, env =self.test_envs, iteration=self.iteration, global_step = self.global_step, seeds=self.dict_test_enviroment['IDseeds'], video_folder=self.dict_test_enviroment['video_folder'], device = self.device)
        mean_reward_OD = test_model(model_A=self.student, model_B = self.Teacher, env =self.test_envs, iteration=self.iteration, global_step = self.global_step, seeds=self.dict_test_enviroment['ODseeds'], video_folder=self.dict_test_enviroment['video_folder'], device = self.device)
        accuracy_ID = self.Tester_under_teacher_ID.TestModel(self.student)
        accuracy_OD = self.Tester_under_teacher_OD.TestModel(self.student)
        wandb.log({
            "test/accuracy_ID_under_T": accuracy_ID,
            "test/accuracy_OD_under_T": accuracy_OD,
            "iteration": self.iteration,
            'global_step': self.global_step
        })
        return mean_reward_ID, mean_reward_OD

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
    ):
        super().__init__(
            student,
            teacher,
            path_folder,
            dict_enviroment,
            device,
            config,
            dict_test_enviroment,
            project_prefix="TeacherDistillation",
        )
        self.Teacher = teacher.to(device)

    def collect_rollouts_aux(self):
        return self.collect_rollouts(self.Teacher)