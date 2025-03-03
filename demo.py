'''
import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

class HighLevelNavigationWrapper(gym.Env):
   
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        # Use .get() to provide a default fixed scaling factor if key is missing.
        self.scaling_factor = pid_params.get('scaling_factor', 0.1)
        
        # The RL agent controls horizontal commands: [roll, pitch, yaw] in range [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Set the observation space identical to the base_env's observation space
        base_obs_space = self.env.observation_space
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32
        )
        
        self.dt = 1.0 / self.env.CTRL_FREQ
        self.error_sum = 0.0
        self.last_error = 0.0
        
    def reset(self, seed=None, options=None):
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, horizontal_action):
        # Get current state from the underlying environment.
        obs = self.env._computeObs()
        obs = np.array(obs, dtype=np.float32)
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target, dtype=np.float32)
        
        # Compute vertical (z-axis) error.
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        max_integral = 5.0
        self.error_sum = np.clip(self.error_sum, -max_integral, max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        
        # Compute thrust using the PID controller.
        base_thrust = 0.265  # Use a fixed value for hover thrust
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt +
            self.Ki * self.error_sum +
            self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        
        # Combine computed thrust with RL horizontal actions to form a full action.
        full_action = np.concatenate(([thrust], horizontal_action))
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)
        return new_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# Create the base Drone environment with the GUI enabled.
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)

# Wrap the base environment with our High-Level Navigation wrapper.
demo_env = HighLevelNavigationWrapper(base_env, pid_params)

# Load the trained RL model.
# Use the model path without the ".zip" extension if Stable Baselines3 appends it automatically.
model_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models/rl_navigation_model"
rl_model = PPO.load(model_path, env=demo_env)

print("Starting demo. Observe the PyBullet GUI for drone behavior.")
obs, info = demo_env.reset()

# Run a controlled loop to step through the simulation.
for step in range(500):
    # Get RL horizontal commands [roll, pitch, yaw] from the trained model.
    horizontal_action, _ = rl_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = demo_env.step(horizontal_action)
    
    demo_env.render()
    
    print(f"Step {step}: Reward = {reward:.3f}, Info = {info}")
    time.sleep(1/30)  # Slow down loop for visualization
    
    if done:
        print("Episode finished; resetting environment...")
        obs, info = demo_env.reset()

demo_env.close()
'''
'''
# demo.py
import os
import time
import numpy as np
import pandas as pd
import csv
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

class HighLevelNavigationWrapper(gym.Env):
    """
    A high-level environment wrapper that uses a PID controller (for low-level vertical
    stability) combined with a reinforcement learning agent (for high-level, horizontal control).
    The wrapper also checks if a waypoint is reached (using an adjustable threshold)
    and automatically commands a new target when reached.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, env, pid_params, waypoint_threshold=0.9):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        self.scaling_factor = pid_params.get('scaling_factor', 0.1)
        # Set the threshold at which the waypoint is considered reached.
        self.waypoint_threshold = waypoint_threshold
        
        # The RL agent controls horizontal commands: [roll, pitch, yaw] in range [-1, 1].
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Set observation space identical to the base environment's.
        base_obs_space = self.env.observation_space
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32
        )
        
        self.dt = 1.0 / self.env.CTRL_FREQ
        self.error_sum = 0.0
        self.last_error = 0.0
        self.max_integral = 5.0  # Integral windup protection

    def reset(self, seed=None, options=None):
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        return np.array(obs, dtype=np.float32), info

    def step(self, horizontal_action):
        # Get the current state from the underlying environment.
        obs = self.env._computeObs()
        obs = np.array(obs, dtype=np.float32)
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target, dtype=np.float32)
        
        # Compute the vertical error (z-axis) and run the PID controller.
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        
        base_thrust = 0.265  # Fixed hover thrust value.
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt +
            self.Ki * self.error_sum +
            self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        
        # Combine vertical thrust computed from PID with RL horizontal actions.
        full_action = np.concatenate(([thrust], horizontal_action))
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)
        
        # Compute the 3D distance error between the drone position and target.
        distance_error = np.linalg.norm(pos - target)
        waypoint_reached = False
        if distance_error < self.waypoint_threshold:
            waypoint_reached = True
            # When a waypoint is reached, generate a new dynamic target.
            self.env.update_target()
        
        # Update info with additional details for logging.
        info.update({
            "distance_error": distance_error,
            "waypoint_reached": waypoint_reached,
            "current_target": target.tolist(),
            "drone_position": pos.tolist()
        })
        
        return new_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def run_demo():
    # Define paths.
    pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
    model_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models/rl_navigation_model"
    log_csv_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/demo_log.csv"
    
    # Load PID parameters from CSV.
    pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()
    
    # Create the base drone environment (with GUI enabled).
    base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
    
    # Wrap the environment with the high-level navigation wrapper.
    demo_env = HighLevelNavigationWrapper(base_env, pid_params, waypoint_threshold=0.9)
    
    # Load the pre-trained RL model (ensure the proper model path).
    rl_model = PPO.load(model_path, env=demo_env)
    
    print("Starting demo. Observe the PyBullet GUI for drone behavior.")
    
    # Open CSV file for logging the demo.
    with open(log_csv_path, "w", newline="") as csvfile:
        fieldnames = ["Step", "Time", "Drone_Position", "Current_Target", "Distance_Error", "Waypoint_Reached", "Reward", "Info"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        start_time = time.time()
        obs, info = demo_env.reset()
        waypoint_reached_flag = False  # Flag to track if any waypoint was reached.
        total_steps = 500
        
        for step in range(total_steps):
            # Retrieve horizontal command from the RL model.
            horizontal_action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(horizontal_action)
            
            current_time = time.time() - start_time
            drone_pos = info.get("drone_position", [])
            current_target = info.get("current_target", [])
            distance_error = info.get("distance_error", None)
            waypoint_reached = info.get("waypoint_reached", False)
            
            # Log this step.
            writer.writerow({
                "Step": step,
                "Time": current_time,
                "Drone_Position": drone_pos,
                "Current_Target": current_target,
                "Distance_Error": distance_error,
                "Waypoint_Reached": waypoint_reached,
                "Reward": reward,
                "Info": info
            })
            csvfile.flush()
            
            # Print a message only when a target waypoint is reached.
            if waypoint_reached:
                print(f"Step {step}: Waypoint reached! Distance error: {distance_error:.2f}")
                waypoint_reached_flag = True
            
            demo_env.render()
            time.sleep(1/30)  # Slow down for visualization.
            
            if done or truncated:
                break
        
        # If no waypoint was reached during the demo, print a message.
        if not waypoint_reached_flag:
            print("No waypoints reached before terminating.")
    
    demo_env.close()
    print(f"Demo finished. Log saved to: {os.path.abspath(log_csv_path)}")

if __name__ == "__main__":
    run_demo()
'''

import os
import time
import numpy as np
import pandas as pd
import csv
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel


class HighLevelNavigationWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        self.scaling_factor = pid_params.get('scaling_factor', 0.1)

        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        base_obs_space = self.env.observation_space
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32
        )

        self.dt = 1.0 / self.env.CTRL_FREQ
        self.error_sum = 0.0
        self.last_error = 0.0
        self.max_integral = 5.0

    def reset(self, seed=None, options=None):
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, horizontal_action):
        obs = self.env._computeObs()
        obs = np.array(obs, dtype=np.float32)
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target, dtype=np.float32)

        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt

        base_thrust = 0.265
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt + self.Ki * self.error_sum + self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)

        full_action = np.concatenate(([thrust], horizontal_action))
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)

        return new_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def run_demo():
    pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
    model_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models/rl_navigation_model"
    log_csv_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/demo_log.csv"

    pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

    # Use demo_mode=False so that training and demo conditions match.
    base_env = DroneNavigationAviary(
        drone_model=DroneModel.CF2X, num_drones=1, gui=True, demo_mode=False
    )
    demo_env = HighLevelNavigationWrapper(base_env, pid_params)
    rl_model = PPO.load(model_path, env=demo_env)

    print("Starting demo. Observe the PyBullet GUI for drone behavior.")

    with open(log_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "Step", "Time", "Drone_Position", "Current_Target", "Distance_Error",
            "Waypoint_Reached", "Reward", "Info"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        start_time = time.time()
        obs, info = demo_env.reset()
        waypoint_reached_flag = False
        total_steps = 500

        for step in range(total_steps):
            horizontal_action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(horizontal_action)

            current_time = time.time() - start_time
            drone_pos = info.get("drone_position", [])
            current_target = info.get("current_target", [])
            distance_error = info.get("tracking_error", None)
            waypoint_reached = info.get("waypoint_reached", False)

            writer.writerow({
                "Step": step,
                "Time": current_time,
                "Drone_Position": drone_pos,
                "Current_Target": current_target,
                "Distance_Error": distance_error,
                "Waypoint_Reached": waypoint_reached,
                "Reward": reward,
                "Info": info
            })
            csvfile.flush()

            if waypoint_reached:
                print(f"Step {step}: Waypoint reached! Distance error: {distance_error:.2f}")
                waypoint_reached_flag = True

            demo_env.render()
            time.sleep(1 / 30)

            if done or truncated:
                print("Episode finished; resetting environment...")
                obs, info = demo_env.reset()

        if not waypoint_reached_flag:
            print("No waypoints reached before terminating.")

    demo_env.close()
    print(f"Demo finished. Log saved to: {os.path.abspath(log_csv_path)}")


if __name__ == "__main__":
    run_demo()
