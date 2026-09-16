from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
)
simulation_app = app_launcher.app

import importlib.metadata
import os

import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from jetbot_rl_navigation.agents.jetbot_ppo_cfg import JetBotPPORunnerCfg
from jetbot_rl_navigation.envs.jetbot_env import JetBotEnv
from jetbot_rl_navigation.envs.jetbot_env_cfg import JetBotEnvCfg


LOG_DIR = "/home/karthik/jetbot_rl_navigation/logs/jetbot_baseline"

CHECKPOINTS = [
    1000,
    1500,
    1800,
    2000,
    2200,
    2400,
    2498,
]

NUM_EPISODES = 200
NUM_ENVS = 8


def evaluate_checkpoint(checkpoint_number):
    """Evaluate one baseline checkpoint."""

    env_cfg = JetBotEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS

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

    checkpoint_path = os.path.join(
        LOG_DIR,
        f"model_{checkpoint_number}.pt",
    )

    print(f"Loading: {checkpoint_path}")

    runner.load(checkpoint_path)

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    base_env = env.unwrapped

    start_successes = base_env.total_successes
    start_collisions = base_env.total_collisions
    start_timeouts = base_env.total_timeouts
    start_episodes = base_env.total_episodes

    obs = env.get_observations()

    episode_lengths = torch.zeros(
        env.num_envs,
        dtype=torch.long,
        device=env.device,
    )

    completed_lengths = []

    while True:
        with torch.inference_mode():
            actions = policy(obs)

        obs, _, dones, _ = env.step(actions)

        episode_lengths += 1

        done_ids = torch.nonzero(
            dones,
            as_tuple=False,
        ).squeeze(-1)

        if done_ids.numel() > 0:
            for idx in done_ids.tolist():
                completed_lengths.append(
                    episode_lengths[idx].item()
                )

            episode_lengths[done_ids] = 0

        completed = base_env.total_episodes - start_episodes

        if completed >= NUM_EPISODES:
            break

    successes = base_env.total_successes - start_successes
    collisions = base_env.total_collisions - start_collisions
    timeouts = base_env.total_timeouts - start_timeouts

    total = successes + collisions + timeouts

    success_rate = 100.0 * successes / total
    collision_rate = 100.0 * collisions / total
    timeout_rate = 100.0 * timeouts / total

    if completed_lengths:
        mean_episode_length = (
            sum(completed_lengths) / len(completed_lengths)
        )
    else:
        mean_episode_length = 0.0

    env.close()

    return {
        "checkpoint": checkpoint_number,
        "episodes": total,
        "success": success_rate,
        "collision": collision_rate,
        "timeout": timeout_rate,
        "mean_length": mean_episode_length,
    }


def main():
    print()
    print("=" * 64)
    print("JETBOT BASELINE CHECKPOINT EVALUATION")
    print("=" * 64)

    results = []

    for checkpoint in CHECKPOINTS:
        print()
        print("-" * 48)
        print(f"Evaluating model_{checkpoint}.pt")
        print("-" * 48)

        result = evaluate_checkpoint(checkpoint)
        results.append(result)

        print(f"Success:   {result['success']:.2f}%")
        print(f"Collision: {result['collision']:.2f}%")
        print(f"Timeout:   {result['timeout']:.2f}%")
        print(f"Mean length: {result['mean_length']:.1f}")
        print(f"Episodes: {result['episodes']}")

    results.sort(
        key=lambda result: (
            -result["success"],
            result["collision"],
            result["timeout"],
            result["mean_length"],
        )
    )

    print()
    print("=" * 64)
    print("FINAL CHECKPOINT RANKING")
    print("=" * 64)

    print(
        f"{'MODEL':<12}"
        f"{'SUCCESS':>10}"
        f"{'COLLISION':>12}"
        f"{'TIMEOUT':>10}"
        f"{'MEAN LEN':>12}"
    )

    print("-" * 56)

    for result in results:
        print(
            f"{result['checkpoint']:<12}"
            f"{result['success']:>9.2f}%"
            f"{result['collision']:>11.2f}%"
            f"{result['timeout']:>9.2f}%"
            f"{result['mean_length']:>12.1f}"
        )

    best = results[0]

    print()
    print("=" * 64)
    print("BEST CHECKPOINT")
    print("=" * 64)

    print(f"model_{best['checkpoint']}.pt")
    print(f"Success rate: {best['success']:.2f}%")
    print(f"Collision rate: {best['collision']:.2f}%")
    print(f"Timeout rate: {best['timeout']:.2f}%")
    print(f"Mean episode length: {best['mean_length']:.1f}")


if __name__ == "__main__":
    main()
    simulation_app.close()