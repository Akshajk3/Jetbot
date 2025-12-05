from webots_rl_test_env import WebotsTestRLEnv
from stable_baselines3 import PPO
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

env = WebotsTestRLEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    device=device
)

# model.learn(total_timesteps=1000)

# model.save("epuck_depth_rl")

obs, info = env.reset()
for _ in range(100):
    env.step((0, 0))

    # env.left_motor.setVelocity(3)
    # env.right_motor.setVelocity(3)