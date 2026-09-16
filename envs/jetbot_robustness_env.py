import omni.usd
import torch

from .jetbot_env import JetBotEnv
from .jetbot_robustness_env_cfg import JetBotRobustnessEnvCfg


class JetBotRobustnessEnv(JetBotEnv):
    """JetBot environment used only for robustness evaluation."""

    cfg: JetBotRobustnessEnvCfg

    def __init__(
        self,
        cfg: JetBotRobustnessEnvCfg,
        render_mode=None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._apply_test_floor_friction()

    def _apply_test_floor_friction(self):
        """Apply a fixed floor-friction perturbation for evaluation."""

        static_friction = self.cfg.robustness_floor_static_friction
        dynamic_friction = self.cfg.robustness_floor_dynamic_friction

        if static_friction is None or dynamic_friction is None:
            return

        stage = omni.usd.get_context().get_stage()

        possible_paths = [
            "/World/ground/physicsMaterial",
            "/World/GroundPlane/physicsMaterial",
            "/physicsScene/defaultMaterial",
        ]

        material_prim = None
        material_path = None

        for path in possible_paths:
            prim = stage.GetPrimAtPath(path)

            if not prim.IsValid():
                continue

            static_attr = prim.GetAttribute("physics:staticFriction")
            dynamic_attr = prim.GetAttribute("physics:dynamicFriction")

            if static_attr.IsValid() and dynamic_attr.IsValid():
                material_prim = prim
                material_path = path
                break

        if material_prim is None:
            raise RuntimeError(
                "Could not find valid ground physics material "
            )

        material_prim.GetAttribute("physics:staticFriction").Set(
            float(static_friction)
        )

        material_prim.GetAttribute("physics:dynamicFriction").Set(
            float(dynamic_friction)
        )

        print(
            "[ROBUSTNESS] Floor material: "
            f"{material_path}"
        )

        print(
            "[ROBUSTNESS] Floor friction: "
            f"static={static_friction}, "
            f"dynamic={dynamic_friction}"
        )

    def _apply_test_wheel_friction(self, env_ids):
        """Apply fixed wheel-friction perturbation for evaluation."""

        friction_value = self.cfg.robustness_wheel_friction

        if friction_value is None:
            return

        friction = torch.full(
            (
                len(env_ids),
                len(self.wheel_joint_ids),
            ),
            float(friction_value),
            dtype=torch.float32,
            device=self.device,
        )

        self.robot.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=friction,
            joint_ids=self.wheel_joint_ids,
            env_ids=env_ids,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._apply_test_wheel_friction(env_ids)

    def _get_observations(self):
        observations = super()._get_observations()

        noise_std = self.cfg.robustness_camera_noise_std

        if noise_std <= 0.0:
            return observations

        images = observations["images"]

        noise = torch.randn_like(images) * noise_std

        observations["images"] = torch.clamp(
            images + noise,
            0.0,
            1.0,
        )

        return observations