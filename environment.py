import pybullet as p
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel
import gym

class SimpleDroneNavigationAviary(BaseAviary):
    def __init__(self, drone_model=DroneModel.CF2X, num_drones=1, gui=True, record=False):
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=10,
            pyb_freq=240,
            ctrl_freq=48,
            physics=True,
            gui=gui,
            record=record
        )
        self.initial_height = 1.0
        self.waypoints = [[0, 0, 1], [3, 0, 1], [3, 3, 1], [0, 3, 1]]  # Simple waypoints
        self.current_waypoint = 0
        self.waypoint_threshold = 0.5
        self._createEnvironment()
        self._createWaypoints()

    def _actionSpace(self):
        return gym.spaces.Box(low=np.zeros(4), high=np.ones(4), dtype=np.float32)

    def _observationSpace(self):
        return gym.spaces.Box(low=np.array([-np.inf]*10), high=np.array([np.inf]*10), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        return np.array([*state[0:3], *state[3:7], *state[10:13]], dtype=np.float32)

    def _createEnvironment(self):
        self.obstacles = []
        pillar_positions = [[2, 2], [-2, -2]]
        for pos in pillar_positions:
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.0)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.0, rgbaColor=[0.5, 0.5, 0.5, 1])
            p.createMultiBody(0, shape, visual, basePosition=[pos[0], pos[1], 1])

    def _createWaypoints(self):
        self.waypoint_ids = []
        colors = [[0, 1, 0, 0.7]] * len(self.waypoints)
        for i, (pos, color) in enumerate(zip(self.waypoints, colors)):
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=color)
            waypoint_id = p.createMultiBody(0, -1, visual, pos)
            self.waypoint_ids.append(waypoint_id)

    def _computeInfo(self):
        """ Provide basic info for the simulation """
        state = self._getDroneStateVector(0)
        return {
            "position": state[0:3],
            "velocity": state[10:13],
            "orientation": state[3:7]
        }

    def reset(self):
        self.current_waypoint = 0
        obs = super().reset()
        init_pos = np.array([0, 0, self.initial_height])
        init_quat = np.array([0., 0., 0., 1.])
        p.resetBasePositionAndOrientation(self.getDroneIds()[0], init_pos, init_quat)
        return self._computeObs()

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        reward = 0.1  # Survival reward
        target_pos = np.array(self.waypoints[self.current_waypoint])
        distance = np.linalg.norm(pos - target_pos)
        reward += 2.0 / (1.0 + distance**2)

        if distance < self.waypoint_threshold:
            reward += 5.0
            self.current_waypoint = min(self.current_waypoint + 1, len(self.waypoints) - 1)

        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        if np.linalg.norm(pos - np.array(self.waypoints[-1])) < self.waypoint_threshold:
            return True
        return False

    def step(self, action):
        action = action * self.MAX_RPM
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            state = self._getDroneStateVector(0)
            p.resetDebugVisualizerCamera(3.0, 45, -30, state[0:3])
        return np.array([])

    def close(self):
        for waypoint_id in self.waypoint_ids:
            p.removeBody(waypoint_id)
        super().close()
