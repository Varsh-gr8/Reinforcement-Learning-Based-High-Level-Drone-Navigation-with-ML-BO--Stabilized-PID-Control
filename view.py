import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

# ------------------------------------------------------------------------------
# High-Level Navigation Wrapper for Demo
# ------------------------------------------------------------------------------
class HighLevelNavigationWrapper:
    """
    A simple minimal wrapper that allows an RL agent to control horizontal motion
    (roll, pitch, yaw) while a pre-tuned PID controller manages altitude.
    """
    def __init__(self, env, pid_params):
        self.env = env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        self.scaling_factor = pid_params['scaling_factor']
        self.dt = 1.0 / self.env.CTRL_FREQ
        self.error_sum = 0.0
        self.last_error = 0.0

    def reset(self):
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        return obs, info

    def step(self, horizontal_action):
        """
        horizontal_action: [roll, pitch, yaw] from the RL agent.
        The PID controller computes thrust for altitude control.
        """
        # Obtain current state.
        obs = self.env._computeObs()
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target)
        # Compute vertical error.
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        max_integral = 5.0
        self.error_sum = np.clip(self.error_sum, -max_integral, max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt

        # Use a base thrust (close to hover, e.g., 0.265 for a 0.027 kg drone).
        base_thrust = 0.265
        thrust = base_thrust + self.scaling_factor * (self.Kp * error_alt +
                                                      self.Ki * self.error_sum +
                                                      self.Kd * error_derivative)
        thrust = np.clip(thrust, 0, 1)
        # Full control action: combine thrust with RL horizontal output.
        full_action = np.concatenate(([thrust], horizontal_action))
        new_obs, reward, done, truncated, info = self.env.step(full_action)
        return new_obs, reward, done, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

# ------------------------------------------------------------------------------
# Load PID Parameters and Create Environment
# ------------------------------------------------------------------------------
pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# Create the base environment with GUI enabled.
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
demo_env = HighLevelNavigationWrapper(base_env, pid_params)

# ------------------------------------------------------------------------------
# Load the Trained RL Model
# ------------------------------------------------------------------------------
# Note: Provide the model path without the '.zip' extension, if Stable Baselines3 appends it automatically.
model_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models/rl_navigation_model"
rl_model = PPO.load(model_path, env=demo_env)

# ------------------------------------------------------------------------------
# Demo Loop
# ------------------------------------------------------------------------------
print("Starting demo. Observe the PyBullet GUI for drone behavior.")
obs, info = demo_env.reset()

for step in range(500):
    # Get RL horizontal commands: [roll, pitch, yaw]
    horizontal_action, _ = rl_model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, info = demo_env.step(horizontal_action)
    demo_env.render()
    
    print(f"Step {step}: Reward = {reward:.3f}, Info = {info}")
    time.sleep(1/30)  # Slow down for visualization
    if done:
        print("Episode finished; resetting environment...")
        obs, info = demo_env.reset()

demo_env.close()
