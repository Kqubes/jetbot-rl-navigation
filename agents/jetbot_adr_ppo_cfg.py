from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


MODEL_CLASS = (
    "jetbot_rl_navigation.agents.jetbot_models:"
    "JetBotTemporalVisionModel"
)


@configclass
class JetBotADRPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the JetBot ADR policy."""

    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 100

    experiment_name = "jetbot_adr"

    obs_groups = {
        "actor": ["policy", "images"],
        "critic": ["policy", "images"],
    }

    actor = RslRlMLPModelCfg(
        class_name=MODEL_CLASS,
        hidden_dims=[512, 256],
        activation="relu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.5,
            std_type="scalar",
        ),
    )

    critic = RslRlMLPModelCfg(
        class_name=MODEL_CLASS,
        hidden_dims=[512, 256],
        activation="relu",
        obs_normalization=False,
        distribution_cfg=None,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=False,
    )