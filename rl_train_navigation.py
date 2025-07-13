import os
import numpy as np
import pandas as pd
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

class HighLevelNavigationWrapper(gym.Env):
    metadata = {"render_modes": ["human"]}
    render_mode = "human"

    def __init__(self, base_env, pid_params):
        super().__init__()
        self.env = base_env
        self.Kp = pid_params.get('Kp', 0.1)
        self.Ki = pid_params.get('Ki', 0.01)
        self.Kd = pid_params.get('Kd', 0.01)
        self.scaling_factor = pid_params.get('scaling_factor', 1.0)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.dt = 1.0 / self.env.CTRL_FREQ
        self.error_sum = 0.0
        self.last_error = 0.0
        self.max_integral = 5.0

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset(seed=seed, options=options)
        return np.array(obs, dtype=np.float32), info

    def step(self, horizontal_action):
        obs = np.array(self.env._computeObs(), dtype=np.float32)
        pos = obs[0:3]
        target = np.array(self.env.current_target, dtype=np.float32)
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        base_thrust = 0.8
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt + self.Ki * self.error_sum + self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        if horizontal_action.ndim > 1:
            horizontal_action = horizontal_action[0]
        full_action = np.concatenate(([thrust], horizontal_action))
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        print(f"[rl_train_navigation.py] Pos: {new_obs[:3]}")
        self.env.render()
        return np.array(new_obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

if __name__ == "__main__":
    print(" Loading PID parameters...")
    pid_params = pd.read_csv("data/best_pid_values.csv").iloc[0].to_dict()
    print(" Creating base environment...")
    base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=False)
    print(" Wrapping with HighLevelNavigationWrapper...")
    rl_env = HighLevelNavigationWrapper(base_env, pid_params)
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rl_navigation_model.zip")
    print(" Initializing PPO agent...")
    model = PPO("MlpPolicy", rl_env, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=model_dir, name_prefix="checkpoint")
    print(" Training started...")
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    print(" Saving model...")
    model.save(model_path)
    rl_env.close()
    print(f" PPO model saved at {model_path}")

