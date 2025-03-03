'''
import os
import numpy as np
import pandas as pd
import gymnasium as gym             # Use gymnasium instead of gym
from gymnasium import spaces        # Use Gymnasium’s spaces
from gymnasium.spaces import Box    # Import Gymnasium's Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

# =============================================================================
# Custom Environment Wrapper for High-Level Navigation using Gymnasium
# =============================================================================
class HighLevelNavigationWrapper(gym.Env):
    """
    Wrapper for DroneNavigationAviary that allows an RL agent to control
    only horizontal motion (roll, pitch, yaw). The vertical (altitude)
    control is handled by a PID controller using tuned parameters.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = base_env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        self.scaling_factor = pid_params['scaling_factor']
        
        # RL agent controls only roll, pitch, yaw: 3-dimensional action space.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Convert the underlying observation space to Gymnasium's Box.
        base_obs_space = self.env.observation_space
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32  # Ensure that the observation space expects float32
        )
        
        # PID state variables for altitude control.
        self.error_sum = 0.0
        self.last_error = 0.0
        self.dt = 1.0 / self.env.CTRL_FREQ
        
        # Integral windup protection
        self.max_integral = 5.0  # Define a maximum integral term to prevent windup
        
    def reset(self, seed=None, options=None):
        # Reset PID states.
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        # Convert observation to float32 (as required by the observation space).
        obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        # RL agent outputs: [roll, pitch, yaw]
        # Get the current observation directly from the underlying environment.
        obs = self.env._computeObs()  # Get the drone's current state.
        obs = np.array(obs, dtype=np.float32)  # Force cast to float32
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target, dtype=np.float32)
        # Compute vertical (z-axis) error.
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)  # Apply integral windup protection
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        
        # Compute thrust using the PID controller.
        base_thrust = 0.265  # Adjust the base thrust value to match the hover thrust calculated based on the drone's weight
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt +
            self.Ki * self.error_sum +
            self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        
        # Combine computed thrust with RL's horizontal action.
        full_action = np.concatenate(([thrust], action))  # Full action: [thrust, roll, pitch, yaw]
        
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)  # Ensure new observation is float32
        return new_obs, reward, terminated, truncated, info  # Return five values as required

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


# =============================================================================
# Load Pre-trained PID Parameters
# =============================================================================
pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# =============================================================================
# Create the Base Environment and Wrap It
# =============================================================================
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
rl_env = HighLevelNavigationWrapper(base_env, pid_params)

# (Optional) Verify your environment with Gymnasium's env checker.
check_env(rl_env, warn=True)

# =============================================================================
# Setup Model Save Directory
# =============================================================================
models_dir = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models"
os.makedirs(models_dir, exist_ok=True)
rl_model_path = os.path.join(models_dir, "rl_navigation_model.zip")

# =============================================================================
# Train the RL Agent Using PPO
# =============================================================================
rl_model = PPO("MlpPolicy", rl_env, verbose=1)
total_timesteps = 200000  # Adjust the timesteps as necessary

rl_model.learn(total_timesteps=total_timesteps)

# Save the trained RL agent.
rl_model.save(rl_model_path)
print("RL Navigation model training complete and saved at:", rl_model_path)

# =============================================================================
# Clean Up
# =============================================================================
rl_env.close()
'''
'''
/\/\/\/\/\/
import os
import numpy as np
import pandas as pd
import gymnasium as gym             # Use gymnasium instead of gym
from gymnasium import spaces        # Use Gymnasium’s spaces
from gymnasium.spaces import Box    # Import Gymnasium's Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

# =============================================================================
# Custom Environment Wrapper for High-Level Navigation using Gymnasium
# =============================================================================
class HighLevelNavigationWrapper(gym.Env):
    """
    Wrapper for DroneNavigationAviary that allows an RL agent to control
    only horizontal motion (roll, pitch, yaw). The vertical (altitude)
    control is handled by a PID controller using tuned parameters.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = base_env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        self.scaling_factor = pid_params['scaling_factor']
        
        # RL agent controls only roll, pitch, yaw: 3-dimensional action space.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Convert the underlying observation space to Gymnasium's Box.
        base_obs_space = self.env.observation_space
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32  # Ensure that the observation space expects float32
        )
        
        # PID state variables for altitude control.
        self.error_sum = 0.0
        self.last_error = 0.0
        self.dt = 1.0 / self.env.CTRL_FREQ
        
        # Integral windup protection
        self.max_integral = 5.0  # Define a maximum integral term to prevent windup
        
    def reset(self, seed=None, options=None):
        # Reset PID states.
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        # Convert observation to float32 (as required by the observation space).
        obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        # RL agent outputs: [roll, pitch, yaw]
        # Get the current observation directly from the underlying environment.
        obs = self.env._computeObs()  # Get the drone's current state.
        obs = np.array(obs, dtype=np.float32)
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target, dtype=np.float32)
        # Compute vertical (z-axis) error.
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        
        # Compute thrust using the PID controller.
        base_thrust = 0.265  # Adjust the base thrust value to match the actual hover thrust.
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt +
            self.Ki * self.error_sum +
            self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        
        # Combine computed thrust with RL's horizontal action.
        full_action = np.concatenate(([thrust], action))
        
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)
        return new_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


# =============================================================================
# Load Pre-trained PID Parameters
# =============================================================================
pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# =============================================================================
# Create the Base Environment and Wrap It
# =============================================================================
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
rl_env = HighLevelNavigationWrapper(base_env, pid_params)

# (Optional) Verify your environment with Gymnasium's env checker.
check_env(rl_env, warn=True)

# =============================================================================
# Setup Model Save Directory
# =============================================================================
models_dir = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models"
os.makedirs(models_dir, exist_ok=True)
rl_model_path = os.path.join(models_dir, "rl_navigation_model.zip")

# =============================================================================
# Train the RL Agent Using PPO
# =============================================================================
print("Starting RL training...")
rl_model = PPO("MlpPolicy", rl_env, verbose=1)
total_timesteps = 200000  # Adjust timesteps as necessary

rl_model.learn(total_timesteps=total_timesteps)

# Save the trained RL agent.
try:
    rl_model.save(rl_model_path)
    print("RL Navigation model training complete and saved at:", os.path.abspath(rl_model_path))
except Exception as e:
    print("Error saving model:", e)

# =============================================================================
# Clean Up
# =============================================================================
rl_env.close()
'''
'''
/\/\/\/\/\/\/\/
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

# =============================================================================
# Custom Environment Wrapper for High-Level Navigation using Gymnasium
# =============================================================================
class HighLevelNavigationWrapper(gym.Env):
    """
    Wrapper for DroneNavigationAviary that allows an RL agent to control
    only horizontal motion (roll, pitch, yaw). The vertical (altitude)
    control is handled by a PID controller using tuned parameters.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, base_env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = base_env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        self.scaling_factor = pid_params['scaling_factor']
        
        # RL agent controls [roll, pitch, yaw]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Convert underlying observation space to Gymnasium's Box.
        base_obs_space = self.env.observation_space
        self.observation_space = Box(
            low=base_obs_space.low,
            high=base_obs_space.high,
            shape=base_obs_space.shape,
            dtype=np.float32
        )
        
        # PID state variables
        self.error_sum = 0.0
        self.last_error = 0.0
        self.dt = 1.0 / self.env.CTRL_FREQ
        
        # Integral windup protection
        self.max_integral = 5.0
        
    def reset(self, seed=None, options=None):
        self.error_sum = 0.0
        self.last_error = 0.0
        obs, info = self.env.reset()
        obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs = self.env._computeObs()
        obs = np.array(obs, dtype=np.float32)
        pos = np.array(obs[0:3])
        target = np.array(self.env.current_target, dtype=np.float32)
        
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        
        base_thrust = 0.265  # Approximate hover thrust for a 0.027-kg drone.
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt +
            self.Ki * self.error_sum +
            self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        
        full_action = np.concatenate(([thrust], action))
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)
        return new_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


# =============================================================================
# Load Pre-trained PID Parameters
# =============================================================================
pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# =============================================================================
# Create Base Environment and Wrap it
# =============================================================================
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
rl_env = HighLevelNavigationWrapper(base_env, pid_params)

# Optionally verify with env checker.
check_env(rl_env, warn=True)

# =============================================================================
# Setup Model Save Directory
# =============================================================================
models_dir = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models"
os.makedirs(models_dir, exist_ok=True)
rl_model_path = os.path.join(models_dir, "rl_navigation_model.zip")

# =============================================================================
# RL Training using PPO with Checkpoint Callback
# =============================================================================
print("Starting RL training...")
rl_model = PPO("MlpPolicy", rl_env, verbose=1)
total_timesteps = 10000  # For testing, try a smaller value first.

# Setup a checkpoint callback to save every 5000 timesteps.
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=models_dir, name_prefix="rl_navigation_model_checkpoint")

try:
    rl_model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted by user.")

finally:
    # Save the final model.
    rl_model.save(rl_model_path)
    print("RL Navigation model training complete and saved at:", os.path.abspath(rl_model_path))

# Clean up
rl_env.close()
'''

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

class HighLevelNavigationWrapper(gym.Env):
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = base_env
        self.pid_params = pid_params
        self.Kp = pid_params['Kp']
        self.Ki = pid_params['Ki']
        self.Kd = pid_params['Kd']
        
        self.scaling_factor = pid_params.get('scaling_factor', 0.1)
         # The RL agent controls horizontal commands: [roll, pitch, yaw] in range [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Set the observation space based on the base environment
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
        # Obtain current state from the underlying environment.
        obs = self.env._computeObs()
        obs = np.array(obs, dtype=np.float32)
        pos = obs[0:3]
        target = np.array(self.env.current_target, dtype=np.float32)
        # Compute vertical (z-axis) error.
        error_alt = target[2] - pos[2]
        self.error_sum += error_alt * self.dt
        self.error_sum = np.clip(self.error_sum, -self.max_integral, self.max_integral)
        error_derivative = (error_alt - self.last_error) / self.dt
        self.last_error = error_alt
        
        # Compute thrust via PID.
        base_thrust = 0.265  # Estimated hover thrust for a 0.027-kg drone.
        thrust = base_thrust + self.scaling_factor * (
            self.Kp * error_alt +
            self.Ki * self.error_sum +
            self.Kd * error_derivative
        )
        thrust = np.clip(thrust, 0, 1)
        
        # Full action: combine PID thrust with horizontal actions.
        full_action = np.concatenate(([thrust], horizontal_action))
        new_obs, reward, terminated, truncated, info = self.env.step(full_action)
        new_obs = np.array(new_obs, dtype=np.float32)
        return new_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# Load PID parameters from CSV.
pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# Create the base environment.
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
# Wrap the environment.
rl_env = HighLevelNavigationWrapper(base_env, pid_params)

# (Optional) Verify environment compliance.
check_env(rl_env, warn=True)

# Setup model save directory.
models_dir = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models"
os.makedirs(models_dir, exist_ok=True)
rl_model_path = os.path.join(models_dir, "rl_navigation_model.zip")

print("Starting RL training...")
rl_model = PPO("MlpPolicy", rl_env, verbose=1)
# For testing, a smaller total timesteps. Increase once verified.
total_timesteps = 10000

# Add a checkpoint callback to save progress every 5000 steps.
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=models_dir, name_prefix="rl_navigation_model_checkpoint")

try:
    rl_model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted by user!")
finally:
    rl_model.save(rl_model_path)
    print("RL Navigation model training complete and saved at:", os.path.abspath(rl_model_path))

rl_env.close()
'''

import os
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import pandas as pd
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

class HighLevelNavigationWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env, pid_params):
        super(HighLevelNavigationWrapper, self).__init__()
        self.env = base_env
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

# Load PID values from CSV
pid_values_path = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/data/best_pid_values.csv"
pid_params = pd.read_csv(pid_values_path).iloc[0].to_dict()

# Initialize base environment
base_env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True, demo_mode=False)
rl_env = HighLevelNavigationWrapper(base_env, pid_params)

# Check environment
check_env(rl_env, warn=True)

# Define RL model directory and save path
models_dir = r"C:/Users/varsh/OneDrive/Documents/Drone_navi/models"
os.makedirs(models_dir, exist_ok=True)
rl_model_path = os.path.join(models_dir, "rl_navigation_model.zip")

print("Starting RL training...")

# Initialize PPO model
rl_model = PPO("MlpPolicy", rl_env, verbose=1)
total_timesteps = 10000

# Set up checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=models_dir,
    name_prefix="rl_navigation_model_checkpoint"
)

# Train RL model
try:
    rl_model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted by user!")
finally:
    rl_model.save(rl_model_path)
    print("RL Navigation model training complete and saved at:", os.path.abspath(rl_model_path))

# Close environment
rl_env.close()
'''