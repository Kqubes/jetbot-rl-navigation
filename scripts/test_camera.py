import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from jetbot_rl_navigation.envs.jetbot_env import JetBotEnv
from jetbot_rl_navigation.envs.jetbot_env_cfg import JetBotEnvCfg

cfg = JetBotEnvCfg()
cfg.scene.num_envs = 1

env = JetBotEnv(cfg=cfg)
obs, _ = env.reset()

print("policy shape:", obs["policy"].shape)
print("images shape:", obs["images"].shape)
print("rgb shape:", env.camera.data.output["rgb"].shape)

actions = torch.tensor([[1.0, 1.0]], device=env.device)

for i in range(300):
    obs, reward, terminated, truncated, info = env.step(actions)

    if i % 25 == 0 or terminated[0] or truncated[0]:
        pos = env.robot.data.root_pos_w[0, :2]
        origin = env.scene.env_origins[0, :2]
        distance = torch.norm(env.goal_xy[0] - pos).item()
        force = env._get_contact_force()[0].item()
        print(
            "step:",
            i,
            "local_x:",
            (pos[0] - origin[0]).item(),
            "local_y:",
            (pos[1] - origin[1]).item(),
            "goal_distance:",
            distance,
            "contact_force:",
            force,
            "terminated:",
            terminated[0].item(),
            "truncated:",
            truncated[0].item(),
        )

env.close()
simulation_app.close()
