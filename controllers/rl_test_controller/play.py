import torch
from webots_rl_test_env import WebotsTestRLEnv
from stable_baselines3 import PPO
import numpy as np

def play():
    env = WebotsTestRLEnv()
    obs, info = env.reset()

    model = PPO.load("epuck_depth_rl.zip", env=env)

    max_steps = 10000

    for step in range(max_steps):
        obs_tensor = np.expand_dims(obs, axis=0)

        action, _states = model.predict(obs, deterministic=True)

        env.left_motor.setVelocity(float(action[0] * 10))
        env.right_motor.setVelocity(float(action[1] * 10))

        env.robot.step(env.timestep)

        obs = env._get_observation()

        if env.is_collision():
            print(f"Collision detected at step {step}, resetting robot...")
            obs, info = env.reset()