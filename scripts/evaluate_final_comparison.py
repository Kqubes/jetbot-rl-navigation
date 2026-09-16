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

from jetbot_rl_navigation.agents.jetbot_adr_ppo_cfg import (
    JetBotADRPPORunnerCfg,
)
from jetbot_rl_navigation.agents.jetbot_ppo_cfg import (
    JetBotPPORunnerCfg,
)
from jetbot_rl_navigation.envs.jetbot_env import JetBotEnv
from jetbot_rl_navigation.envs.jetbot_env_cfg import JetBotEnvCfg


MODELS = [
    {
        "name": "Baseline",
        "checkpoint": (
            "/home/karthik/jetbot_rl_navigation/logs/"
            "jetbot_baseline/model_1500.pt"
        ),
        "config": "baseline",
    },
    {
        "name": "ADR_Wheel_Camera",
        "checkpoint": (
            "/home/karthik/jetbot_rl_navigation/logs/"
            "jetbot_adr/model_1500.pt"
        ),
        "config": "adr",
    },
    {
        "name": "ADR_Wheel_Camera_Floor",
        "checkpoint": (
            "/home/karthik/jetbot_rl_navigation/logs/"
            "jetbot_adr/model_2800.pt"
        ),
        "config": "adr",
    },
]

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
    """Set all random seeds used during evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_agent_cfg(config_type):
    """Create the correct PPO configuration for a model."""

    if config_type == "baseline":
        agent_cfg = JetBotPPORunnerCfg()
    else:
        agent_cfg = JetBotADRPPORunnerCfg()

    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")

    return handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        rsl_rl_version,
    )


env_cfg = JetBotEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.seed = TEST_SEEDS[0]

env = JetBotEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

base_env = env.unwrapped


def evaluate_seed(policy, seed):
    """Evaluate one policy on one unseen environment seed."""

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
        mean_length = (
            sum(completed_lengths) / len(completed_lengths)
        )
    else:
        mean_length = 0.0

    return {
        "seed": seed,
        "episodes": total,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "mean_length": mean_length,
    }


def evaluate_model(model_info):
    """Evaluate one trained model across all unseen test seeds."""

    agent_cfg = get_agent_cfg(
        model_info["config"]
    )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=base_env.device,
    )

    runner.load(
        model_info["checkpoint"]
    )

    policy = runner.get_inference_policy(
        device=base_env.device
    )

    print()
    print("=" * 56)
    print(f"EVALUATING: {model_info['name']}")
    print("=" * 56)

    seed_results = []

    for seed in TEST_SEEDS:
        result = evaluate_seed(
            policy,
            seed,
        )

        seed_results.append(result)

        print(
            f"Seed {seed:<5} | "
            f"S={result['success_rate']:6.2f}% | "
            f"C={result['collision_rate']:6.2f}% | "
            f"T={result['timeout_rate']:6.2f}% | "
            f"Len={result['mean_length']:7.1f}"
        )

    total_successes = sum(
        result["successes"]
        for result in seed_results
    )

    total_collisions = sum(
        result["collisions"]
        for result in seed_results
    )

    total_timeouts = sum(
        result["timeouts"]
        for result in seed_results
    )

    total_episodes = (
        total_successes
        + total_collisions
        + total_timeouts
    )

    success_rate = (
        100.0 * total_successes / total_episodes
    )

    collision_rate = (
        100.0 * total_collisions / total_episodes
    )

    timeout_rate = (
        100.0 * total_timeouts / total_episodes
    )

    mean_length = (
        sum(
            result["mean_length"] * result["episodes"]
            for result in seed_results
        )
        / total_episodes
    )

    seed_success_rates = [
        result["success_rate"]
        for result in seed_results
    ]

    return {
        "name": model_info["name"],
        "checkpoint": model_info["checkpoint"],
        "episodes": total_episodes,
        "success": success_rate,
        "collision": collision_rate,
        "timeout": timeout_rate,
        "mean_length": mean_length,
        "success_std": float(
            np.std(seed_success_rates)
        ),
        "best_seed": max(seed_success_rates),
        "worst_seed": min(seed_success_rates),
    }


def main():
    print()
    print("=" * 64)
    print("FINAL BASELINE VS ADR COMPARISON")
    print("=" * 64)

    print(f"Seeds: {len(TEST_SEEDS)}")
    print(f"Episodes per seed: {EPISODES_PER_SEED}")
    print(
        f"Episodes per model: "
        f"{len(TEST_SEEDS) * EPISODES_PER_SEED}"
    )

    results = [
        evaluate_model(model_info)
        for model_info in MODELS
    ]

    print()
    print("=" * 80)
    print("FINAL NOMINAL UNSEEN RESULTS")
    print("=" * 80)

    print(
        f"{'MODEL':<28}"
        f"{'SUCCESS':>10}"
        f"{'COLLISION':>12}"
        f"{'TIMEOUT':>10}"
        f"{'STD':>9}"
        f"{'MEAN LEN':>12}"
    )

    print("-" * 81)

    for result in results:
        print(
            f"{result['name']:<28}"
            f"{result['success']:>9.2f}%"
            f"{result['collision']:>11.2f}%"
            f"{result['timeout']:>9.2f}%"
            f"{result['success_std']:>8.2f}%"
            f"{result['mean_length']:>12.1f}"
        )

    print()
    print("=" * 80)
    print("SEED ROBUSTNESS")
    print("=" * 80)

    for result in results:
        print(
            f"{result['name']}: "
            f"best={result['best_seed']:.2f}% | "
            f"worst={result['worst_seed']:.2f}%"
        )

    best = max(
        results,
        key=lambda result: (
            result["success"],
            -result["collision"],
            -result["timeout"],
        ),
    )

    print()
    print("=" * 80)
    print("BEST NOMINAL MODEL")
    print("=" * 80)

    print(f"Model: {best['name']}")
    print(f"Success: {best['success']:.2f}%")
    print(f"Collision: {best['collision']:.2f}%")
    print(f"Timeout: {best['timeout']:.2f}%")
    print(
        f"Success std: "
        f"{best['success_std']:.2f}%"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()