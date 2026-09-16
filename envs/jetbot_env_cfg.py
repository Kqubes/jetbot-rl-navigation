from __future__ import annotations

from dataclasses import replace

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


JETBOT_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
)


@configclass
class JetBotEnvCfg(DirectRLEnvCfg):
    """Configuration for the JetBot vision-based navigation environment."""

    # Environment
    decimation = 4
    episode_length_s = 60.0

    action_space = 2
    observation_space = 3
    state_space = 0

    # Simulation
    sim = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    scene = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=15.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # JetBot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=JETBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_wheel_joint",
                    "right_wheel_joint",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=5.0,
                stiffness=0.0,
                damping=2.0,
            ),
        },
    )

    # Obstacles
    _base_obstacle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle1",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
                metallic=0.2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 2.0, 0.2),
        ),
    )

    obstacle1 = _base_obstacle
    obstacle2 = replace(
        _base_obstacle,
        prim_path="/World/envs/env_.*/Obstacle2",
    )
    obstacle3 = replace(
        _base_obstacle,
        prim_path="/World/envs/env_.*/Obstacle3",
    )
    obstacle4 = replace(
        _base_obstacle,
        prim_path="/World/envs/env_.*/Obstacle4",
    )
    obstacle5 = replace(
        _base_obstacle,
        prim_path="/World/envs/env_.*/Obstacle5",
    )

    # Camera
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/chassis/vision_camera",
        update_period=0.0,
        height=64,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.12, 0.0, 0.18),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Navigation
    history_len = 3
    action_scale = 5.0
    target_reach_threshold = 0.4
    collision_distance = 0.32