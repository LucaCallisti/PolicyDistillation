import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

class RenderFrameWrapper(Wrapper):
    """Adds an `env_frame` (RGB array) to info on reset/step."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Prime the wrapper to determine frame shape for the observation space.
        obs, _ = self.env.reset()
        frame = self._render_frame()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=frame.shape,
            dtype=frame.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._render_frame()
        info = info or {}
        info["state_observation"] = obs
        info["frame_observation"] = frame
        return frame, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._render_frame()
        info = info or {}
        info["state_observation"] = obs
        info["frame_observation"] = frame
        return frame, reward, terminated, truncated, info

    def _render_frame(self):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                f"Underlying env returned None on render."
            )
        return frame
    

class RenderFrameWrapper_CarRacing(Wrapper):
    """Adds an `env_frame` (RGB array) to info on reset/step."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Prime the wrapper to determine frame shape for the observation space.
        pass
        

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = info or {}
        info["state_observation"] = None
        info["frame_observation"] = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = info or {}
        info["state_observation"] = None
        info["frame_observation"] = obs
        return obs, reward, terminated, truncated, info


class Float32ObsWrapper(Wrapper):
    """Casts observations (reset/step) to float32."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Box):
            low = env.observation_space.low.astype(np.float32, copy=False)
            high = env.observation_space.high.astype(np.float32, copy=False)
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=env.observation_space.shape,
                dtype=np.float32,
            )
        else:
            raise TypeError("Float32ObsWrapper supports only Box observation spaces.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['state_observation'] = self._cast(obs)
        return self._cast(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['state_observation'] = self._cast(obs)
        return self._cast(obs), reward, terminated, truncated, info

    @staticmethod
    def _cast(obs):
        return obs.astype("float32", copy=False)
    

class AddStateObsWrapper(Wrapper):
    """Adds state observation to info on reset/step."""
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = info or {}
        info["state_observation"] = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = info or {}
        info["state_observation"] = obs
        return obs, reward, terminated, truncated, info
