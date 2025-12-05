from webots_rl_test_env import WebotsTestRLEnv
from stable_baselines3 import PPO
import torch

def train():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
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

    model.learn(total_timesteps=1000)

    model.save("epuck_depth_rl")

    print("Finished Training")