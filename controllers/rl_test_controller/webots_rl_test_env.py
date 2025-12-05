import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Robot, RangeFinder, Motor, Supervisor

class WebotsTestRLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timestep=32):
        super().__init__()

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep()) if timestep is None else timestep

        self.left_motor: Motor = self.robot.getDevice('left wheel motor')
        self.right_motor: Motor = self.robot.getDevice('right wheel motor')
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        self.range_finder: RangeFinder = self.robot.getDevice('range-finder')
        self.range_finder.enable(self.timestep)

        w = self.range_finder.getWidth()
        h = self.range_finder.getHeight()

        self.current_step = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(w * h,),
            dtype=np.float32
        )

        self.MAX_SPEED = 6.28
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_SPEED, -self.MAX_SPEED], dtype=np.float32),
            high=np.array([self.MAX_SPEED, self.MAX_SPEED], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.robot.simulationReset()
        self.robot.step(self.timestep)

        print("resetting env")

        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        EPS = 1e-3  # small buffer
        left_speed = 3
        right_speed = 3
        self.left_motor.setVelocity(float(left_speed))
        self.right_motor.setVelocity(float(right_speed))
        self.robot.step(self.timestep)
        self.current_step += 1

        obs = self._get_observation()
        reward = self.compute_reward(obs, action)

        terminated = reward < -5.0
        truncated = False
        info = {}

        # Debug print
        print(f"Step {self.current_step}: action={action}, reward={reward}, min_dist={np.min(obs)}")

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        w = self.range_finder.getWidth()
        h = self.range_finder.getHeight()

        img = np.array(self.range_finder.getRangeImage(), dtype=np.float32)
        img = np.clip(img, 0.0, 10.0)
        depth = img.reshape((h, w))

        return depth.flatten()

    def compute_reward(self, depth, action):
        min_dist = np.min(depth)
        if min_dist < 0.1:
            return -10.0  # collision

        # reward for moving forward (average wheel speed in forward direction)
        forward_reward = (action[0] + action[1]) / 2.0
        # combine forward movement with obstacle avoidance
        reward = min_dist + 0.5 * forward_reward
        return reward