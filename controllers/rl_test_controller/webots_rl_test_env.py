import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import RangeFinder, Motor, Supervisor, DistanceSensor

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

        self.ir_sensors = []
        for i in range(8):
            sensor: DistanceSensor = self.robot.getDevice(f'ps{i}')
            sensor.enable(self.timestep)
            self.ir_sensors.append(sensor)

        w = self.range_finder.getWidth()
        h = self.range_finder.getHeight()

        self.current_step = 0

        self.max_steps = 500

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
        # self.current_step = 0
        self.robot.step(self.timestep)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        print("resetting env")

        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        left_speed, right_speed = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        self.left_motor.setVelocity(float(left_speed))
        self.right_motor.setVelocity(float(right_speed))
        self.robot.step(self.timestep)
        self.current_step += 1

        obs = self._get_observation()
        reward = self.compute_reward(obs, action)

        terminated = self.is_collision()
        truncated = False
        info = {}

        print(f"Step {self.current_step}: action={action}, reward={reward}, min_dist={np.min(obs)}")

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        w = self.range_finder.getWidth()
        h = self.range_finder.getHeight()

        img = np.array(self.range_finder.getRangeImage(), dtype=np.float32)
        img = np.clip(img, 0.0, 10.0)
        depth = img.reshape((h, w))

        return depth.flatten()

    def is_collision(self):
        threshold = 160.0
        for sensor in self.ir_sensors:
            if sensor.getValue() > threshold:
                return True
        return False

    def compute_reward(self, depth, action):
        if self.is_collision():
            return -10.0

        min_dist = np.min(depth)
        min_dist = min(min_dist, 0.5)

        forward_reward = (action[0] + action[1]) / 2.0
        straight_reward = -abs(action[0] - action[1])
        reward = min_dist + 2.0 * forward_reward + 0.2 * straight_reward
        return reward