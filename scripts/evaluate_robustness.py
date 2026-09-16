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
from jetbot_rl_navigation.envs.jetbot_robustness_env import (
    JetBotRobustnessEnv,
)
from jetbot_rl_navigation.envs.jetbot_robustness_env_cfg import (
    JetBotRobustnessEnvCfg,
)


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

TESTS = [
    {
        "name": "Camera_Noise",
        "camera_noise": 0.05,
        "wheel_friction": None,
        "floor_static": None,
        "floor_dynamic": None,
    },
    {
        "name": "Wheel_Friction",
        "camera_noise": 0.0,
        "wheel_friction": 0.20,
        "floor_static": None,
        "floor_dynamic": None,
    },
    {
        "name": "Floor_Friction",
        "camera_noise": 0.0,
        "wheel_friction": None,
        "floor_static": 0.80,
        "floor_dynamic": 0.70,
    },
    {
        "name": "Combined",
        "camera_noise": 0.05,
        "wheel_friction": 0.20,
        "floor_static": 0.80,
        "floor_dynamic": 0.70,
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

EPISODES_PER_SEED = 10
NUM_ENVS = 8


def set_test_seed(seed):
    """Set random seeds for reproducible robustness evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_agent_cfg(config_type):
    """Return the PPO configuration used by the selected model."""

    if config_type == "baseline":
        agent_cfg = JetBotPPORunnerCfg()
    else:
        agent_cfg = JetBotADRPPORunnerCfg()

    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")

    return handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        rsl_rl_version,
    )


def evaluate_model_condition(model_info, test_info):
    """Evaluate one model under one robustness condition."""

    env_cfg = JetBotRobustnessEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = TEST_SEEDS[0]

    env_cfg.robustness_camera_noise_std = test_info["camera_noise"]
    env_cfg.robustness_wheel_friction = test_info["wheel_friction"]

    env_cfg.robustness_floor_static_friction = test_info[
        "floor_static"
    ]
    env_cfg.robustness_floor_dynamic_friction = test_info[
        "floor_dynamic"
    ]

    env = JetBotRobustnessEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    base_env = env.unwrapped
    agent_cfg = get_agent_cfg(model_info["config"])

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=base_env.device,
    )

    runner.load(model_info["checkpoint"])

    policy = runner.get_inference_policy(
        device=base_env.device
    )

    total_successes = 0
    total_collisions = 0
    total_timeouts = 0

    all_lengths = []
    seed_success_rates = []

    for seed in TEST_SEEDS:
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

        total_successes += successes
        total_collisions += collisions
        total_timeouts += timeouts

        all_lengths.extend(completed_lengths)

        seed_success = 100.0 * successes / total
        seed_success_rates.append(seed_success)

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

    if all_lengths:
        mean_length = sum(all_lengths) / len(all_lengths)
    else:
        mean_length = 0.0

    success_std = float(
        np.std(seed_success_rates)
    )

    result = {
        "model": model_info["name"],
        "test": test_info["name"],
        "episodes": total_episodes,
        "success": success_rate,
        "collision": collision_rate,
        "timeout": timeout_rate,
        "length": mean_length,
        "std": success_std,
        "worst_seed": min(seed_success_rates),
        "best_seed": max(seed_success_rates),
    }

    env.close()

    return result


def main():
    print()
    print("=" * 80)
    print("FINAL ROBUSTNESS EVALUATION")
    print("=" * 80)

    print(f"Models: {len(MODELS)}")
    print(f"Conditions: {len(TESTS)}")
    print(f"Seeds per condition: {len(TEST_SEEDS)}")
    print(
        f"Target episodes/model/condition: "
        f"{len(TEST_SEEDS) * EPISODES_PER_SEED}"
    )

    results = []

    for test_info in TESTS:
        print()
        print("=" * 80)
        print(f"TEST CONDITION: {test_info['name']}")
        print("=" * 80)

        for model_info in MODELS:
            print(f"Running {model_info['name']}...")

            result = evaluate_model_condition(
                model_info,
                test_info,
            )

            results.append(result)

            print(
                f"Success={result['success']:.2f}% | "
                f"Collision={result['collision']:.2f}% | "
                f"Timeout={result['timeout']:.2f}% | "
                f"Std={result['std']:.2f}%"
            )

    print()
    print("=" * 96)
    print("FINAL ROBUSTNESS RESULTS")
    print("=" * 96)

    print(
        f"{'TEST':<18}"
        f"{'MODEL':<28}"
        f"{'SUCCESS':>10}"
        f"{'COLLISION':>12}"
        f"{'TIMEOUT':>10}"
        f"{'STD':>9}"
        f"{'WORST':>9}"
    )

    print("-" * 96)

    for result in results:
        print(
            f"{result['test']:<18}"
            f"{result['model']:<28}"
            f"{result['success']:>9.2f}%"
            f"{result['collision']:>11.2f}%"
            f"{result['timeout']:>9.2f}%"
            f"{result['std']:>8.2f}%"
            f"{result['worst_seed']:>8.2f}%"
        )

    print()
    print("=" * 80)
    print("AVERAGE ROBUSTNESS ACROSS ALL PERTURBATIONS")
    print("=" * 80)

    for model_info in MODELS:
        model_results = [
            result
            for result in results
            if result["model"] == model_info["name"]
        ]

        average_success = float(
            np.mean(
                [
                    result["success"]
                    for result in model_results
                ]
            )
        )

        average_collision = float(
            np.mean(
                [
                    result["collision"]
                    for result in model_results
                ]
            )
        )

        average_timeout = float(
            np.mean(
                [
                    result["timeout"]
                    for result in model_results
                ]
            )
        )

        print(
            f"{model_info['name']:<28} | "
            f"Success={average_success:6.2f}% | "
            f"Collision={average_collision:6.2f}% | "
            f"Timeout={average_timeout:6.2f}%"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()