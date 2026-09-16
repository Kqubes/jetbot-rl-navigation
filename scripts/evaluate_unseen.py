from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
)
simulation_app = app_launcher.app

import importlib.metadata
import random

import numpy as np
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
    "jetbot_baseline/model_1500.pt"
)

TEST_SEEDS = [
    1001,
    2002,
    3003,
    4004,
    5005,
    6006,
    7007,
    8008,
    9009,
    10010,
]

EPISODES_PER_SEED = 20
NUM_ENVS = 8


def set_test_seed(seed):
    """Set random seeds for evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


env_cfg = JetBotEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.seed = TEST_SEEDS[0]

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

base_env = env.unwrapped


def run_seed(seed):
    """Evaluate the baseline policy for one unseen seed."""

    set_test_seed(seed)
    env.reset()

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

        if completed >= EPISODES_PER_SEED:
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

    return {
        "seed": seed,
        "episodes": total,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "mean_episode_length": mean_episode_length,
    }


def main():
    print()
    print("=" * 56)
    print("BASELINE UNSEEN ENVIRONMENT EVALUATION")
    print("=" * 56)

    print("Checkpoint: model_1500.pt")
    print(f"Number of test seeds: {len(TEST_SEEDS)}")
    print(f"Target episodes per seed: {EPISODES_PER_SEED}")

    results = []

    for seed in TEST_SEEDS:
        print()
        print("-" * 48)
        print(f"Testing seed {seed}")
        print("-" * 48)

        result = run_seed(seed)
        results.append(result)

        print(f"Episodes:  {result['episodes']}")
        print(f"Success:   {result['success_rate']:.2f}%")
        print(f"Collision: {result['collision_rate']:.2f}%")
        print(f"Timeout:   {result['timeout_rate']:.2f}%")
        print(
            f"Mean length: "
            f"{result['mean_episode_length']:.1f}"
        )

    total_successes = sum(
        result["successes"]
        for result in results
    )

    total_collisions = sum(
        result["collisions"]
        for result in results
    )

    total_timeouts = sum(
        result["timeouts"]
        for result in results
    )

    total_episodes = (
        total_successes
        + total_collisions
        + total_timeouts
    )

    overall_success = (
        100.0 * total_successes / total_episodes
    )

    overall_collision = (
        100.0 * total_collisions / total_episodes
    )

    overall_timeout = (
        100.0 * total_timeouts / total_episodes
    )

    weighted_length_sum = sum(
        result["mean_episode_length"]
        * result["episodes"]
        for result in results
    )

    mean_length = (
        weighted_length_sum / total_episodes
    )

    success_rates = [
        result["success_rate"]
        for result in results
    ]

    print()
    print("=" * 56)
    print("FINAL UNSEEN BASELINE RESULTS")
    print("=" * 56)

    print("Checkpoint: model_1500.pt")
    print(f"Total episodes: {total_episodes}")
    print(
        f"Overall success rate: "
        f"{overall_success:.2f}%"
    )
    print(
        f"Overall collision rate: "
        f"{overall_collision:.2f}%"
    )
    print(
        f"Overall timeout rate: "
        f"{overall_timeout:.2f}%"
    )
    print(
        f"Mean episode length: "
        f"{mean_length:.1f}"
    )
    print(
        f"Mean seed success: "
        f"{np.mean(success_rates):.2f}%"
    )
    print(
        f"Success std across seeds: "
        f"{np.std(success_rates):.2f}%"
    )
    print(
        f"Best seed success: "
        f"{max(success_rates):.2f}%"
    )
    print(
        f"Worst seed success: "
        f"{min(success_rates):.2f}%"
    )

    print()
    print("=" * 56)
    print("PER-SEED RESULTS")
    print("=" * 56)

    for result in results:
        print(
            f"Seed {result['seed']:<5} | "
            f"E={result['episodes']:<3} | "
            f"S={result['success_rate']:6.2f}% | "
            f"C={result['collision_rate']:6.2f}% | "
            f"T={result['timeout_rate']:6.2f}% | "
            f"Len={result['mean_episode_length']:7.1f}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()