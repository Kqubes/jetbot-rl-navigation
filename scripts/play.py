import argparse

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a trained JetBot navigation policy."
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments.",
    )

    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


args_cli = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata

import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from jetbot_rl_navigation.agents.jetbot_ppo_cfg import JetBotPPORunnerCfg
from jetbot_rl_navigation.envs.jetbot_env import JetBotEnv
from jetbot_rl_navigation.envs.jetbot_env_cfg import JetBotEnvCfg


CHECKPOINT = (
    "/home/karthik/jetbot_rl_navigation/logs/"
    "jetbot_adr/model_1500.pt"
)


def main():
    env_cfg = JetBotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = JetBotPPORunnerCfg()

    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        rsl_rl_version,
    )

    env = JetBotEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=env.unwrapped.device,
    )

    runner.load(CHECKPOINT)

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    obs = env.get_observations()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()