import gymnasium as gym
import os



def make_env(dict, idx, wrappers = []):       
    def thunk():
        if (idx in dict["render_idx"]) or (idx in dict["record_video_idx"]):
            if dict['env_name'] == "LunarLander":
                from Enviroment.My_env import my_LunarLander
                env = my_LunarLander(render_mode='rgb_array')
            elif dict['env_name'] == "Pusher-v5":
                # env = gym.make(dict['env_name'], render_mode='rgb_array', default_camera_config={
                #                 "distance": 3.5,
                #                 "lookat": (0.0, 0.0, 0.0),
                #                 "azimuth": 270,
                #                 "elevation": -35,
                #             })
                env = gym.make(dict['env_name'], render_mode='rgb_array', default_camera_config={
                                    "distance": 2.5,
                                    "lookat": (0.0, 0.2, 0.0),
                                    "azimuth": 270,
                                    "elevation": -90,
                                })
            else:   
                env = gym.make(dict['env_name'], render_mode='rgb_array')
            if idx in dict["record_video_idx"]:
                print(f"Recording videos for env index {idx}")
                dict['video_folder'] = os.path.join(dict["folder_path"], "videos", dict["run_name"])
                env = gym.wrappers.RecordVideo(env,  dict['video_folder'], episode_trigger=lambda episode_id: True)
        else:
            if dict['env_name'] == "LunarLander":
                env = my_LunarLander()
            else:
                env = gym.make(dict['env_name'])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    return thunk


