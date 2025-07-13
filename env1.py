import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

CID = None

def get_pybullet_cid():
    global CID
    if CID is None:
        CID = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return CID

class DroneNavigationAviary(BaseAviary):
    metadata = {"render_modes": ["human"]}
    render_mode = "human"

    def __init__(self, drone_model=DroneModel.CF2X, num_drones=1, gui=True, demo_mode=False):
        self.cid = get_pybullet_cid()
        self.gui = gui
        self.demo_mode = demo_mode
        self.waypoint_threshold = 0.9 if demo_mode else 0.2
        self.CTRL_FREQ = 48
        self.noise_std = 0.15
        self.coupling_factor = 0.01
        self._np_random = None
        self.sim_iter = 0

        self.urdf_path = os.path.join(os.getcwd(), "gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found at {self.urdf_path}")

        super().__init__(drone_model=drone_model, num_drones=num_drones, gui=gui)
        self.reset()

    def _actionSpace(self):
        return spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def _setup_simulation(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        self.DRONE_IDS = [p.loadURDF(self.urdf_path, [0, 0, 1], physicsClientId=self.cid)]
        p.changeDynamics(self.DRONE_IDS[0], -1, linearDamping=0.05, angularDamping=0.05, physicsClientId=self.cid)
        self._load_obstacles()

    def _load_obstacles(self):
        self.OBSTACLES = []
        positions = [(1, 1, 0.5), (-1, -1, 0.5), (0, 2, 0.5), (-2, 0, 0.5)]
        for pos in positions:
            self.OBSTACLES.append(p.loadURDF("sphere2.urdf", pos, physicsClientId=self.cid))

    def sanitize_target(self, target):
        lower_bounds = np.array([-10, -10, 0], dtype=np.float32)
        upper_bounds = np.array([10, 10, 10], dtype=np.float32)
        return np.clip(np.array(target, dtype=np.float32), lower_bounds, upper_bounds)

    def generate_random_target(self):
        target = np.random.uniform(low=[-5, -5, 0.5], high=[5, 5, 2])
        return self.sanitize_target(target)

    def _computeObs(self):
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.cid)
        vel, ang_vel = p.getBaseVelocity(self.DRONE_IDS[0], physicsClientId=self.cid)
        self._last_vel = vel
        self._last_ang_vel = ang_vel
        obs = np.concatenate([np.array(pos), np.array(quat), np.array(vel)])
        return obs[:10]

    def _computeReward(self, obs):
        current_pos = np.array(obs[:3])
        dist = np.linalg.norm(current_pos - self.current_target)
        return 50.0 - dist if dist < self.waypoint_threshold else -dist

    def _computeInfo(self):
        return {"target": self.current_target.tolist()}

    def _checkTermination(self, obs):
        pos = np.array(obs[:3])
        return np.any(np.abs(pos[:2]) > 10) or pos[2] < 0 or pos[2] > 10

    def step(self, action):
        action = np.clip(action, 0, 1)
        max_thrust = 20.0
        max_torque = 0.05
        thrust = action[0] * max_thrust * 2.0
        roll_cmd = (action[1] - 0.5) * 2.5 * max_torque
        pitch_cmd = (action[2] - 0.5) * 2.5 * max_torque
        yaw_cmd = (action[3] - 0.5) * 2.5 * max_torque
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.cid)
        rot_matrix = p.getMatrixFromQuaternion(quat)
        body_z = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]])
        force = thrust * body_z
        torque = np.array([roll_cmd, pitch_cmd, yaw_cmd])
        p.applyExternalTorque(self.DRONE_IDS[0], -1, torque, flags=p.LINK_FRAME, physicsClientId=self.cid)
        p.applyExternalTorque(self.DRONE_IDS[0], -1, torque, flags=p.LINK_FRAME, physicsClientId=self.cid)
        p.stepSimulation(physicsClientId=self.cid)
        self.sim_iter += 1
        obs = self._computeObs()
        reward = self._computeReward(obs)
        collision = any(p.getContactPoints(self.DRONE_IDS[0], obs_id, physicsClientId=self.cid) for obs_id in self.OBSTACLES)
        if collision:
            reward -= 10
            print("Collision detected!")
        tracking_error = np.linalg.norm(np.array(obs[:3]) - self.current_target)
        info = {
            "tracking_error": tracking_error,
            "drone_position": obs[:3].tolist(),
            "current_target": self.current_target.tolist(),
            "waypoint_reached": tracking_error < self.waypoint_threshold
        }
        terminated = self._checkTermination(obs)
        truncated = False
        if info["waypoint_reached"]:
            print(" Waypoint reached. Generating a new target...")
            self.update_target()
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass  # GUI already active

    def update_target(self):
        self.current_target = self.generate_random_target()
        print(f" New dynamic target: {self.current_target}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        if not p.isConnected(self.cid):
            self.cid = p.connect(p.GUI)
        p.resetSimulation(physicsClientId=self.cid)
        self._setup_simulation()
        self.current_target = self.generate_random_target()
        self.sim_iter = 0
        obs = self._computeObs()
        info = self._computeInfo()
        return obs, info
