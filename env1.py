'''
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

class DroneNavigationAviary(BaseAviary):
    """
    Custom environment for drone navigation and PID optimization.
    
    This version uses a dynamic target for exploration.
    Each time the drone reaches the current target, a new random target is generated.
    """

    def __init__(self, drone_model=DroneModel.CF2X, num_drones=1, gui=True):
        # Override action and observation spaces.
        self._actionSpace = lambda: spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self._observationSpace = lambda: spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        super().__init__(drone_model=drone_model, num_drones=num_drones, gui=gui)

        self.gui = gui
        # Set URDF path.
        self.urdf_path = os.path.join(os.getcwd(), "gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"❌ URDF file not found at {self.urdf_path}")

        # Use a dynamic target (instead of fixed waypoints).
        self.current_target = self.generate_random_target()
        self.waypoint_threshold = 0.2  # If error is below this, the target is considered achieved.
        self.CTRL_FREQ = 48

        # RPM signal improvements.
        self.noise_std = 0.15
        self.coupling_factor = 0.01

        self._setup_simulation()
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        self.reset()

    def sanitize_target(self, target):
        """
        Convert target to a NumPy array (float32) and clip values.
        Assumes acceptable x, y ∈ [-10, 10] and z ∈ [0, 10].
        """
        target = np.array(target, dtype=np.float32)
        lower_bounds = np.array([-10, -10, 0], dtype=np.float32)
        upper_bounds = np.array([10, 10, 10], dtype=np.float32)
        return np.clip(target, lower_bounds, upper_bounds)

    def generate_random_target(self):
        """
        Generate a random target within safe exploration boundaries.
        For example, choose x, y ∈ [-5, 5] and z ∈ [0.5, 2].
        """
        new_target = np.random.uniform(low=[-5, -5, 0.5], high=[5, 5, 2])
        return self.sanitize_target(new_target)

    def _setup_simulation(self):
        """Initialize PyBullet, load the drone, and load obstacles."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.DRONE_IDS = [p.loadURDF(self.urdf_path, [0, 0, 1])]
        self._load_obstacles()

    def _load_obstacles(self):
        """Load static obstacles."""
        self.OBSTACLES = []
        obstacle_positions = [(1, 1, 0.5), (-1, -1, 0.5), (0, 2, 0.5), (-2, 0, 0.5)]
        for pos in obstacle_positions:
            obstacle = p.loadURDF("sphere2.urdf", pos)
            self.OBSTACLES.append(obstacle)

    def _computeObs(self):
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        vel, _ = p.getBaseVelocity(self.DRONE_IDS[0])
        full_obs = np.concatenate([np.array(pos), np.array(quat), np.array(vel)])
        return full_obs[:10]

    def compute_motor_rpms(self, thrust, roll, pitch, yaw):
        """
        Compute motor RPMs using an X-mixer with cross-axis coupling and noise.
        Returns a 4-element vector clipped to [0, 1].
        """
        rpm_1 = thrust + roll - pitch + yaw + self.coupling_factor * pitch
        rpm_2 = thrust - roll - pitch - yaw + self.coupling_factor * roll
        rpm_3 = thrust - roll + pitch + yaw + self.coupling_factor * pitch
        rpm_4 = thrust + roll + pitch - yaw + self.coupling_factor * roll
        rpms = np.array([rpm_1, rpm_2, rpm_3, rpm_4])
        noise = np.random.normal(0, self.noise_std, rpms.shape)
        self.last_noise = noise
        rpms_noisy = rpms + noise
        return np.clip(rpms_noisy, 0, 1)

    def step(self, action):
        """
        Apply the control action (normalized thrust, roll, pitch, yaw) to the drone.

        Mappings:
          • Thrust: [0, 1] —> [0, 7.5] N.
          • Roll/Pitch/Yaw: [0, 1] —> [-5, +5] N·m (with 0.5 being neutral).

        Only a collision will flag termination.
        Reaching the target (if error is below threshold) updates the dynamic target.
        """
        action = np.clip(action, 0, 1)
        max_thrust = 7.5
        max_torque = 5.0
        thrust = action[0] * max_thrust
        roll_cmd = (action[1] - 0.5) * 2 * max_torque
        pitch_cmd = (action[2] - 0.5) * 2 * max_torque
        yaw_cmd = (action[3] - 0.5) * 2 * max_torque

        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        rot_matrix = p.getMatrixFromQuaternion(quat)
        body_z = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]])
        force = thrust * body_z
        p.applyExternalForce(self.DRONE_IDS[0], -1, force, pos, p.WORLD_FRAME)
        torque = np.array([roll_cmd, pitch_cmd, yaw_cmd])
        p.applyExternalTorque(self.DRONE_IDS[0], -1, torque, p.WORLD_FRAME)

        p.stepSimulation()
        if self.gui:
            time.sleep(1/240)
        obs = self._computeObs()
        reward = self._computeReward(obs)
        done = self._checkTermination(obs)

        current_pos = np.array(obs[:3])
        # If the drone has reached (within threshold) the current target, update it.
        if np.linalg.norm(current_pos - self.current_target) < self.waypoint_threshold:
            self.update_target()

        tracking_error = np.linalg.norm(current_pos - self.current_target)
        info = {
            "target_position": self.current_target.tolist(),
            "tracking_error": float(tracking_error),
            "disturbance": self.last_noise.tolist() if hasattr(self, "last_noise") else [0, 0, 0, 0]
        }
        return obs, reward, done, False, info

    def _computeReward(self, obs):
        current_pos = np.array(obs[:3])
        distance = np.linalg.norm(current_pos - self.current_target)
        reward = -distance
        if distance < self.waypoint_threshold:
            reward += 10
        return reward

    def _checkTermination(self, obs):
        for obs_id in self.OBSTACLES:
            if p.getContactPoints(self.DRONE_IDS[0], obs_id):
                return True
        return False

    def update_target(self):
        """Generate a new random target for exploration."""
        self.current_target = self.generate_random_target()
        print(f"New dynamic target: {self.current_target}")

    def reset(self):
        """Reset the simulation and reinitialize state (including a new dynamic target)."""
        p.resetSimulation()
        self._setup_simulation()
        self.current_target = self.generate_random_target()
        return self._computeObs(), {}

if __name__ == "__main__":
    # For debugging: run a short simulation loop.
    env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
    obs, _ = env.reset()
    for _ in range(100):
        action = np.array([0.5, 0.5, 0.5, 0.5])
        obs, reward, done, _, info = env.step(action)
        print("Obs:", obs, "Reward:", reward, "Info:", info)
        if done:
            print("Collision detected. Resetting.")
            obs, _ = env.reset()
    env.close()
'''
'''
/\/\/\/\/\/\/\
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

class DroneNavigationAviary(BaseAviary):
    """
    Custom environment for drone navigation and PID optimization.
    This version uses a dynamic target for exploration.
    Each time the drone reaches the current target, a new random target is generated.
    """
    def __init__(self, drone_model=DroneModel.CF2X, num_drones=1, gui=True):
        # Override action and observation spaces.
        self._actionSpace = lambda: spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self._observationSpace = lambda: spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        super().__init__(drone_model=drone_model, num_drones=num_drones, gui=gui)

        self.gui = gui

        # Set the URDF path.
        self.urdf_path = os.path.join(os.getcwd(), "gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"❌ URDF file not found at {self.urdf_path}")

        # Initialize dynamic target.
        self.current_target = self.generate_random_target()
        self.waypoint_threshold = 0.2  # When the tracking error falls below this, the target is considered reached.
        self.CTRL_FREQ = 48

        # Parameters for RPM signal improvements.
        self.noise_std = 0.15
        self.coupling_factor = 0.01

        self._setup_simulation()
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        self.reset()

    def sanitize_target(self, target):
        """
        Convert target to a float32 NumPy array and clip within acceptable bounds.
        Acceptable bounds: x, y in [-10, 10] and z in [0, 10].
        """
        target = np.array(target, dtype=np.float32)
        lower_bounds = np.array([-10, -10, 0], dtype=np.float32)
        upper_bounds = np.array([10, 10, 10], dtype=np.float32)
        return np.clip(target, lower_bounds, upper_bounds)

    def generate_random_target(self):
        """
        Generate a random target within safe exploration boundaries.
        For example, choose x, y ∈ [-5, 5] and z ∈ [0.5, 2].
        """
        new_target = np.random.uniform(low=[-5, -5, 0.5], high=[5, 5, 2])
        return self.sanitize_target(new_target)

    def _setup_simulation(self):
        """
        Initialize PyBullet: set search path, gravity, load drone, and obstacles.
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.DRONE_IDS = [p.loadURDF(self.urdf_path, [0, 0, 1])]
        self._load_obstacles()

    def _load_obstacles(self):
        """
        Load static obstacles into the simulation.
        """
        self.OBSTACLES = []
        obstacle_positions = [(1, 1, 0.5), (-1, -1, 0.5), (0, 2, 0.5), (-2, 0, 0.5)]
        for pos in obstacle_positions:
            obstacle = p.loadURDF("sphere2.urdf", pos)
            self.OBSTACLES.append(obstacle)

    def _computeObs(self):
        """
        Compute the observation by concatenating the drone's position,
        orientation (quaternion), and linear velocity. We only return the first 10 elements.
        """
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        vel, _ = p.getBaseVelocity(self.DRONE_IDS[0])
        full_obs = np.concatenate([np.array(pos), np.array(quat), np.array(vel)])
        return full_obs[:10]

    def compute_motor_rpms(self, thrust, roll, pitch, yaw):
        """
        Compute motor RPMs using an X-mixer with cross-axis coupling and noise.
        Returns a 4-element vector with values clipped between 0 and 1.
        """
        rpm_1 = thrust + roll - pitch + yaw + self.coupling_factor * pitch
        rpm_2 = thrust - roll - pitch - yaw + self.coupling_factor * roll
        rpm_3 = thrust - roll + pitch + yaw + self.coupling_factor * pitch
        rpm_4 = thrust + roll + pitch - yaw + self.coupling_factor * roll
        rpms = np.array([rpm_1, rpm_2, rpm_3, rpm_4])
        noise = np.random.normal(0, self.noise_std, rpms.shape)
        self.last_noise = noise
        rpms_noisy = rpms + noise
        return np.clip(rpms_noisy, 0, 1)

    def step(self, action):
        """
        Apply the control action (normalized thrust, roll, pitch, yaw) to the drone.
        Mappings:
          • Thrust: [0, 1]  → [0, 7.5] N.
          • Roll/Pitch/Yaw: [0, 1]  → [-5, +5] N·m (with 0.5 as neutral).
        """
        action = np.clip(action, 0, 1)
        max_thrust = 7.5
        max_torque = 5.0
        thrust = action[0] * max_thrust
        roll_cmd = (action[1] - 0.5) * 2 * max_torque
        pitch_cmd = (action[2] - 0.5) * 2 * max_torque
        yaw_cmd = (action[3] - 0.5) * 2 * max_torque

        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        rot_matrix = p.getMatrixFromQuaternion(quat)
        body_z = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]])
        force = thrust * body_z
        p.applyExternalForce(self.DRONE_IDS[0], -1, force, pos, p.WORLD_FRAME)
        torque = np.array([roll_cmd, pitch_cmd, yaw_cmd])
        p.applyExternalTorque(self.DRONE_IDS[0], -1, torque, p.WORLD_FRAME)

        p.stepSimulation()
        if self.gui:
            time.sleep(1/240)
        obs = self._computeObs()
        reward = self._computeReward(obs)
        done = self._checkTermination(obs)

        # Compute tracking error between drone's position and the current target.
        current_pos = np.array(obs[:3])
        tracking_error = np.linalg.norm(current_pos - self.current_target)
        print(f"Drone position: {current_pos}, Target: {self.current_target}, Tracking error: {tracking_error:.3f}")
        if tracking_error < self.waypoint_threshold:
            print("Waypoint reached. Generating a new target...")
            self.update_target()

        info = {
            "target_position": self.current_target.tolist(),
            "tracking_error": float(tracking_error),
            "disturbance": self.last_noise.tolist() if hasattr(self, "last_noise") else [0, 0, 0, 0]
        }
        return obs, reward, done, False, info

    def _computeReward(self, obs):
        current_pos = np.array(obs[:3])
        distance = np.linalg.norm(current_pos - self.current_target)
        reward = -distance
        if distance < self.waypoint_threshold:
            reward += 10
        return reward

    def _checkTermination(self, obs):
        for obs_id in self.OBSTACLES:
            if p.getContactPoints(self.DRONE_IDS[0], obs_id):
                print("Collision detected with obstacle.")
                return True
        return False

    def update_target(self):
        """Generate a new random target for exploration."""
        self.current_target = self.generate_random_target()
        print(f"New dynamic target: {self.current_target}")

    def reset(self):
        """
        Reset the simulation and generate a new dynamic target.
        This resets PyBullet's simulation, loads the drone and obstacles again.
        """
        p.resetSimulation()
        self._setup_simulation()
        self.current_target = self.generate_random_target()
        return self._computeObs(), {}

if __name__ == "__main__":
    # Debug simulation loop.
    env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
    obs, _ = env.reset()
    for _ in range(100):
        action = np.array([0.5, 0.5, 0.5, 0.5])
        obs, reward, done, _, info = env.step(action)
        print("Obs:", obs, "Reward:", reward, "Info:", info)
        if done:
            print("Collision detected. Resetting.")
            obs, _ = env.reset()
    env.close()
'''
# env1.py
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

class DroneNavigationAviary(BaseAviary):
    """
    Custom environment for drone navigation and PID optimization.
    This version uses a dynamic target for exploration. Each time the drone reaches the current target, a new random target is generated.
    The 'demo_mode' flag lets you modify the waypoint threshold (e.g., to 0.9) for demo purposes.
    For consistency (and to match training), use demo_mode=False.
    """
    def __init__(self, drone_model=DroneModel.CF2X, num_drones=1, gui=True, demo_mode=False):
        # Override action and observation spaces.
        self._actionSpace = lambda: spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self._observationSpace = lambda: spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        super().__init__(drone_model=drone_model, num_drones=num_drones, gui=gui)
        self.gui = gui

        # Set the URDF path.
        self.urdf_path = os.path.join(os.getcwd(), "gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"❌ URDF file not found at {self.urdf_path}")

        # Initialize dynamic target.
        self.current_target = self.generate_random_target()
        # Use 0.9 if demo_mode is True; for consistency with training, use demo_mode=False (waypoint threshold=0.2)
        self.waypoint_threshold = 0.9 if demo_mode else 0.2
        self.CTRL_FREQ = 48

        # Parameters for RPM signal improvements.
        self.noise_std = 0.15
        self.coupling_factor = 0.01

        self._setup_simulation()
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        self.reset()

    def sanitize_target(self, target):
        target = np.array(target, dtype=np.float32)
        lower_bounds = np.array([-10, -10, 0], dtype=np.float32)
        upper_bounds = np.array([10, 10, 10], dtype=np.float32)
        return np.clip(target, lower_bounds, upper_bounds)

    def generate_random_target(self):
        new_target = np.random.uniform(low=[-5, -5, 0.5], high=[5, 5, 2])
        return self.sanitize_target(new_target)

    def _setup_simulation(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.DRONE_IDS = [p.loadURDF(self.urdf_path, [0, 0, 1])]
        self._load_obstacles()

    def _load_obstacles(self):
        self.OBSTACLES = []
        obstacle_positions = [(1, 1, 0.5), (-1, -1, 0.5), (0, 2, 0.5), (-2, 0, 0.5)]
        for pos in obstacle_positions:
            obstacle = p.loadURDF("sphere2.urdf", pos)
            self.OBSTACLES.append(obstacle)

    def _computeObs(self):
        if not p.isConnected():
            mode = p.GUI if self.gui else p.DIRECT
            print("Warning: Physics server disconnected. Reconnecting in mode:", mode)
            p.connect(mode)
            self._setup_simulation()
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        vel, _ = p.getBaseVelocity(self.DRONE_IDS[0])
        full_obs = np.concatenate([np.array(pos, dtype=np.float32),
                                   np.array(quat, dtype=np.float32),
                                   np.array(vel, dtype=np.float32)])
        return full_obs[:10]

    def compute_motor_rpms(self, thrust, roll, pitch, yaw):
        rpm_1 = thrust + roll - pitch + yaw + self.coupling_factor * pitch
        rpm_2 = thrust - roll - pitch - yaw + self.coupling_factor * roll
        rpm_3 = thrust - roll + pitch + yaw + self.coupling_factor * pitch
        rpm_4 = thrust + roll + pitch - yaw + self.coupling_factor * roll
        rpms = np.array([rpm_1, rpm_2, rpm_3, rpm_4], dtype=np.float32)
        noise = np.random.normal(0, self.noise_std, rpms.shape)
        self.last_noise = noise
        rpms_noisy = rpms + noise
        return np.clip(rpms_noisy, 0, 1)

    def step(self, action):
        action = np.clip(action, 0, 1)
        max_thrust = 7.5
        max_torque = 5.0
        thrust = action[0] * max_thrust
        roll_cmd = (action[1] - 0.5) * 2 * max_torque
        pitch_cmd = (action[2] - 0.5) * 2 * max_torque
        yaw_cmd = (action[3] - 0.5) * 2 * max_torque

        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        rot_matrix = p.getMatrixFromQuaternion(quat)
        body_z = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]], dtype=np.float32)
        force = thrust * body_z
        p.applyExternalForce(self.DRONE_IDS[0], -1, force, pos, p.WORLD_FRAME)
        torque = np.array([roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
        p.applyExternalTorque(self.DRONE_IDS[0], -1, torque, p.WORLD_FRAME)

        p.stepSimulation()
        if self.gui:
            time.sleep(1/240)
        obs = self._computeObs()
        reward = self._computeReward(obs)

        # Check for collisions with obstacles.
        collision_detected = False
        for obs_id in self.OBSTACLES:
            if p.getContactPoints(self.DRONE_IDS[0], obs_id):
                collision_detected = True
                print("Collision detected with obstacle.")
                break

        # Adjust collision penalty (try –10 instead of –20)
        if collision_detected:
            reward -= 10

        # Do not terminate the episode due to collisions during training/demo.
        done = False

        current_pos = np.array(obs[:3], dtype=np.float32)
        tracking_error = np.linalg.norm(current_pos - self.current_target)
        info = {
            "tracking_error": tracking_error,
            "drone_position": current_pos.tolist(),
            "current_target": self.current_target.tolist(),
            "disturbance": self.last_noise.tolist() if hasattr(self, "last_noise") else [0, 0, 0, 0]
        }
        if tracking_error < self.waypoint_threshold:
            info["waypoint_reached"] = True
            print("Waypoint reached. Generating a new target...")
            self.update_target()
        else:
            info["waypoint_reached"] = False

        return obs, float(reward), done, False, info

    def _computeReward(self, obs):
        current_pos = np.array(obs[:3], dtype=np.float32)
        distance = np.linalg.norm(current_pos - self.current_target)
        # Increase the bonus when within threshold (e.g., +50 instead of +10)
        if distance < self.waypoint_threshold:
            reward = -distance + 50
        else:
            reward = -distance
        return float(reward)

    def _checkTermination(self, obs):
        # We no longer terminate due to collisions.
        return False

    def update_target(self):
        self.current_target = self.generate_random_target()
        print(f"New dynamic target: {self.current_target}")

    def reset(self):
        p.resetSimulation()
        self._setup_simulation()
        self.current_target = self.generate_random_target()
        return self._computeObs(), {}

if __name__ == "__main__":
    # Debug simulation loop.
    env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True, demo_mode=True)
    obs, _ = env.reset()
    for _ in range(100):
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        obs, reward, done, _, info = env.step(action)
        print("Obs:", obs, "Reward:", reward, "Info:", info)
        if done:
            print("Collision detected. Resetting.")
            obs, _ = env.reset()
    env.close()
