import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

# Load PID values
pid_values_path = r"C:\Users\varsh\OneDrive\Documents\Drone_navi\data\best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()
Kp = pid_params['Kp']
Ki = pid_params['Ki']
Kd = pid_params['Kd']
scaling_factor = pid_params['scaling_factor']

print("Loaded PID values:")
print(f"  Kp: {Kp}\n  Ki: {Ki}\n  Kd: {Kd}\n  Scaling Factor: {scaling_factor}")

# Initialize Environment
env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
obs, _ = env.reset()

# Load RL Model with corrected path (only one .zip extension)
rl_model_path = r"C:\Users\varsh\OneDrive\Documents\Drone_navi\models\rl_navigation_model.zip" 
rl_model = PPO.load(rl_model_path, env=env)

dt = 1.0 / env.CTRL_FREQ
max_steps = 200

# PID Control Variables for Altitude
error_sum = 0.0
last_error = 0.0

# Control Loop
for step in range(max_steps):
    pos = np.array(obs[0:3])           # Drone position: [x, y, z]
    target = np.array(env.current_target)  # Target waypoint: [x, y, z]

    # Compute Altitude Error
    error_alt = target[2] - pos[2]
    error_sum += error_alt * dt
    error_derivative = (error_alt - last_error) / dt
    last_error = error_alt

    # PID Thrust Control
    thrust = 0.5 + scaling_factor * (Kp * error_alt + Ki * error_sum + Kd * error_derivative)
    thrust = np.clip(thrust, 0, 1)

    # RL Model for Horizontal Control
    rl_action, _ = rl_model.predict(obs, deterministic=True)
    action = np.concatenate(([thrust], rl_action))

    obs, reward, done, _, info = env.step(action)

    print(f"Step {step}: Position = {pos}, Target = {target}, Altitude Error = {error_alt:.3f}, "
          f"Thrust = {thrust:.3f}, Reward = {reward:.3f}")

    if done:
        print("Target reached or episode ended.")
        break

env.close()
