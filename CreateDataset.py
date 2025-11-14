import argparse
import torch
from Utils_PPO import PPO_Agent, set_seeds, Mean_start_end, get_screen
import gymnasium as gym
import os
import matplotlib.pyplot as plt
from My_env import LunarLander  # Import your modified LunarLander environment

class TensorDataset():
    def __init__(self, capacity):
        capacity = capacity + 1000
        self.frames = torch.zeros(capacity, 133, 200, dtype=torch.float32)
        self.logprobs = torch.zeros(capacity, 4, dtype=torch.float32)
        self.states = torch.zeros(capacity, 8, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.new_episode_index = []
        self.seed_run = []
        self.current_index = 0

    def push_episode(self, frames, states, logprobs, reward, seed_run):
        self.seed_run.append(seed_run)
        self.new_episode_index.append(self.current_index)
        length = len(frames)
        self.frames[self.current_index:self.current_index + length] = torch.stack(frames).squeeze()
        self.states[self.current_index:self.current_index + length] = torch.stack(states).squeeze()
        self.logprobs[self.current_index:self.current_index + length] = torch.stack(logprobs).squeeze()
        self.rewards[self.current_index:self.current_index + length] = torch.tensor(reward, dtype=torch.float32)
        self.current_index += length

    def save(self, path):
        seed_run_tensor = torch.tensor(self.seed_run, dtype=torch.int64)
        new_episode_index_tensor = torch.tensor(self.new_episode_index, dtype=torch.int64)
        dict = {
            'frames': self.frames[:self.current_index],
            'states': self.states[:self.current_index],
            'logprobs': self.logprobs[:self.current_index],
            'seed_run': seed_run_tensor,
            'new_episode_index': new_episode_index_tensor,
            'rewards': self.rewards[:self.current_index]
        }
        torch.save(dict, path)
    
    def __len__(self):
        return self.current_index
        

def collect_episodes(model, num_data=100):
    device = next(model.parameters()).device
    env = LunarLander(render_mode='rgb_array', random_initial_state=True, enable_wind=False, apply_initial_random=False)

    dataset = TensorDataset(capacity=num_data)
    get_screen_fun = get_screen()
    
    env.reset(seed=args.seed + 1000)  
    env.action_space.seed(args.seed + 1000)

    model.eval()

    i=0
    while len(dataset) < num_data:
        i += 1
        Episodes_state, Episodes_frame_state, Episodes_logprobs_Teacher, Reward = [], [], [], []

        seed = i + 200
        state = env.reset(seed= seed)[0]
        env.action_space.seed(seed)
        
        frame = get_screen_fun.get_screen(env)
        torch_frame = torch.as_tensor(frame, dtype=torch.float32, device=device)
        
        done = False
        ep_reward = 0
        steps_done = 0
        landed_exp = 5
        while not done:

            steps_done += 1      
            state = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)

            with torch.no_grad():
                action, _, _, _, logprob = model.get_action_and_value(state, logprob_bool=True)
            if not (state[0][-1] == 1 and state[0][-2] == 1):
                Episodes_logprobs_Teacher.append(logprob)
                Episodes_frame_state.append(torch_frame)
                Episodes_state.append(state)
            if (state[0][-1] == 1 and state[0][-2] == 1) and landed_exp > 0:
                landed_exp -= 1
                Episodes_logprobs_Teacher.append(logprob)
                Episodes_frame_state.append(torch_frame)
                Episodes_state.append(state)

            state, reward, terminated, truncated, _ = env.step(action.cpu().item())
            if steps_done > 1000:
                print(f"Episode {i+1} is too long, terminating.")
                reward = -100  # Penalize for too long episodes
                terminated = True
            done = terminated or truncated
            if len(Episodes_state) > len(Reward):
                Reward.append(reward)
            ep_reward += reward

            frame = get_screen_fun.get_screen(env)
            torch_frame = torch.as_tensor(frame, dtype=torch.float32, device=device)
                

        if args.check_episodes:
            if ep_reward > 200:
                print(f"Episode {i+1} with reward {ep_reward:.2f} is above threshold, saving data. current dataset size: {len(dataset)}")
                dataset.push_episode(Episodes_frame_state, Episodes_state, Episodes_logprobs_Teacher, Reward, seed_run=seed)
            else:
                print(f"Episode {i+1} with reward {ep_reward:.2f} is below threshold, not saving data. current dataset size: {len(dataset)}")
        else:
            print(f"Episode {i+1} with reward {ep_reward:.2f}")
            dataset.push_episode(Episodes_frame_state, Episodes_state, Episodes_logprobs_Teacher, Reward, seed_run=seed)


    env.close()

    save_path = os.path.join(args.path, f"PPO_LunarLander_Dataset_{i}ep_{len(dataset)}elt.pth")
    dataset.save(save_path)
    print(f"Dataset saved to {save_path}")


def save_sample_frames(frames, save_dir, prefix="frame"):   
    for idx, frame in enumerate(frames):
        if frame.is_cuda:
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame.numpy()
        
        frame_np = frame_np.squeeze(0).squeeze(0)  

        save_path = os.path.join(save_dir, f"{prefix}_{idx}.png")
        plt.imsave(save_path, frame_np, cmap='gray' if frame_np.ndim == 2 else None)
        
    print(f"Saved {len(frames)} frames to {save_dir}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunardLander Collector")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--Model-path', type=str, default="/home/l.callisti/Distillation_LunarLander/PPO/Teacher_files/Teacher_lr_1.00e-04_ent_coef_1.00e-02_Anneal_ent_ppo_agent.pt")
    parser.add_argument('--path', type=str, default="/home/l.callisti/Distillation_LunarLander/PPO/Datasets")
    parser.add_argument('--device', type=str, default='cuda:5' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--Dataset_size', type=int, default=100000, help='Size of the dataset to collect')
    parser.add_argument('--check_episodes', action='store_true', default=True, help='Check the collected dataset')

    args = parser.parse_args()


    set_seeds(args.seed)

    model_weights = torch.load(args.Model_path, map_location=args.device, weights_only=True)
    model = PPO_Agent(envs=LunarLander()).to(args.device)
    model.load_state_dict(model_weights)
    collect_episodes(model, num_data=args.Dataset_size)

