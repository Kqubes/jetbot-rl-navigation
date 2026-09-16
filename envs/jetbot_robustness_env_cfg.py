from isaaclab.utils import configclass

from .jetbot_env_cfg import JetBotEnvCfg


@configclass
class JetBotRobustnessEnvCfg(JetBotEnvCfg):
    """Configuration for JetBot robustness evaluation."""

    robustness_camera_noise_std = 0.0
    robustness_wheel_friction = None

    robustness_floor_static_friction = None
    robustness_floor_dynamic_friction = None