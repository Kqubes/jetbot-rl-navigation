import argparse
import importlib.metadata
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train the JetBot baseline navigation policy."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata
import os

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from jetbot_rl_navigation.agents.jetbot_ppo_cfg import JetBotPPORunnerCfg
from jetbot_rl_navigation.envs.jetbot_env import JetBotEnv
from jetbot_rl_navigation.envs.jetbot_env_cfg import JetBotEnvCfg


LOG_DIR = "/home/karthik/jetbot_rl_navigation/logs/jetbot_baseline"


def main():
    env_cfg = JetBotEnvCfg()
    agent_cfg = JetBotPPORunnerCfg()

    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        rsl_rl_version,
    )

    env = JetBotEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=LOG_DIR,
        device=env.unwrapped.device,
    )

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=False,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()