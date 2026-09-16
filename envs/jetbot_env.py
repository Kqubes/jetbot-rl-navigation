from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .jetbot_env_cfg import JetBotEnvCfg


def define_navigation_markers():
    """The forward-heading and goal-direction arrows."""

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/NavigationArrows",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0),
                ),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        },
    )

    return VisualizationMarkers(cfg=marker_cfg)


def define_goal_marker():
    """The goal-position marker."""

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.20,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            ),
        },
    )

    return VisualizationMarkers(marker_cfg)


class JetBotEnv(DirectRLEnv):
    """Vision-based JetBot navigation environment with obstacle avoidance."""

    cfg: JetBotEnvCfg

    def __init__(
        self,
        cfg: JetBotEnvCfg,
        render_mode=None,
        **kwargs,
    ):
        self._camera_hist = None
        self.history_len = cfg.history_len
        self.num_obstacles = 5

        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.camera = self.scene["camera"]

        self.obstacles = [
            self.scene[f"obstacle{i}"]
            for i in range(1, self.num_obstacles + 1)
        ]

        self.wheel_joint_ids, self.wheel_joint_names = self.robot.find_joints(
            [
                "left_wheel_joint",
                "right_wheel_joint",
            ],
            preserve_order=True,
        )

        self.target_pos = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )

        self.prev_dist = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.up_dir = torch.tensor(
            [0.0, 0.0, 1.0],
            device=self.device,
        )

        self.actions = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )

        self.arrows = define_navigation_markers()
        self.arrows.set_visibility(True)

        self.goal_marker = define_goal_marker()
        self.goal_marker.set_visibility(True)

        self.total_episodes = 0
        self.total_successes = 0
        self.total_collisions = 0
        self.total_timeouts = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.camera = Camera(self.cfg.camera)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["camera"] = self.camera

        self.obstacles = []

        for i in range(1, self.num_obstacles + 1):
            obstacle_cfg = getattr(self.cfg, f"obstacle{i}")
            obstacle = RigidObject(obstacle_cfg)

            self.obstacles.append(obstacle)
            self.scene.rigid_objects[f"obstacle{i}"] = obstacle

        spawn_ground_plane(
            prim_path="/World/GroundPlane",
            cfg=GroundPlaneCfg(),
        )

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _get_goal_vec(self):
        return self.target_pos - self.robot.data.root_pos_w.torch

    def _pre_physics_step(self, actions: torch.Tensor):
        self.raw_actions = actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

        velocity_targets = self.actions * self.cfg.action_scale

        self.robot.set_joint_velocity_target(
            velocity_targets,
            joint_ids=self.wheel_joint_ids,
        )

        self.extras.setdefault("log", {}).update(
            {
                "Actions/raw_left": self.raw_actions[:, 0].mean(),
                "Actions/raw_right": self.raw_actions[:, 1].mean(),
                "Actions/raw_abs": self.raw_actions.abs().mean(),
                "Actions/clamped_left": self.actions[:, 0].mean(),
                "Actions/clamped_right": self.actions[:, 1].mean(),
                "Actions/clamped_abs": self.actions.abs().mean(),
            }
        )

        self.goal_marker.visualize(self.target_pos)
        self._visualize_arrows()

    def _apply_action(self):
        pass

    def _visualize_arrows(self):
        goal_vec = self._get_goal_vec()

        command_yaws = torch.atan2(
            goal_vec[:, 1],
            goal_vec[:, 0],
        )

        command_rot = math_utils.quat_from_angle_axis(
            command_yaws,
            self.up_dir,
        )

        robot_rot = self.robot.data.root_quat_w.torch

        locations = self.robot.data.root_pos_w.torch.clone()
        locations[:, 2] += 0.5

        marker_locations = torch.cat(
            (locations, locations),
            dim=0,
        )

        marker_rotations = torch.cat(
            (robot_rot, command_rot),
            dim=0,
        )

        marker_indices = torch.cat(
            (
                torch.zeros(
                    self.num_envs,
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.ones(
                    self.num_envs,
                    dtype=torch.int32,
                    device=self.device,
                ),
            ),
            dim=0,
        )

        self.arrows.visualize(
            translations=marker_locations,
            orientations=marker_rotations,
            marker_indices=marker_indices,
        )

    def _get_observations(self):
        camera_data = self.camera.data.output["rgb"].float() / 255.0

        if self._camera_hist is None:
            self._camera_hist = (
                camera_data.unsqueeze(1)
                .repeat(
                    1,
                    self.history_len,
                    1,
                    1,
                    1,
                )
                .contiguous()
            )
        else:
            new_frame = camera_data.unsqueeze(1)

            self._camera_hist = torch.cat(
                [
                    self._camera_hist[:, 1:],
                    new_frame,
                ],
                dim=1,
            )

        goal_vec = self._get_goal_vec()

        goal_dist = torch.linalg.norm(
            goal_vec,
            dim=-1,
            keepdim=True,
        )

        unit_goal = goal_vec / (goal_dist + 1e-6)

        state_input = torch.hstack(
            (
                unit_goal[:, :2],
                goal_dist,
            )
        )

        rgb_frames = self._camera_hist.permute(
            0,
            1,
            4,
            2,
            3,
        )

        n, t, c, h, w = rgb_frames.shape

        images = rgb_frames.reshape(
            n,
            t * c,
            h,
            w,
        )

        return {
            "policy": state_input,
            "images": images,
        }

    def _get_collision(self):
        robot_pos = self.robot.data.root_pos_w.torch[:, :2]

        collision = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        for obstacle in self.obstacles:
            obstacle_pos = obstacle.data.root_pos_w[:, :2]

            distance = torch.norm(
                robot_pos - obstacle_pos,
                dim=-1,
            )

            collision |= distance < self.cfg.collision_distance

        return collision

    def _get_rewards(self):
        robot_lin_vel = self.robot.data.root_lin_vel_w.torch[:, :3]
        goal_vec = self._get_goal_vec()

        goal_dir = goal_vec / (
            torch.norm(
                goal_vec,
                dim=-1,
                keepdim=True,
            )
            + 1e-6
        )

        velocity_proj = torch.sum(
            robot_lin_vel * goal_dir,
            dim=-1,
        )

        progress_reward = velocity_proj * 2.5

        dist = torch.linalg.norm(
            goal_vec,
            dim=-1,
        )

        dist_delta = self.prev_dist - dist
        self.prev_dist = dist.clone()

        dist_reward = dist_delta * 0.5

        backward_act = torch.sum(
            torch.clamp(
                self.actions,
                max=0.0,
            ),
            dim=1,
        )

        backward_penalty = backward_act * 0.5

        collision = self._get_collision()
        collision_reward = collision.float() * -5.0

        reached = dist < self.cfg.target_reach_threshold
        success_reward = reached.float() * 100.0

        robot_quat = self.robot.data.root_quat_w.torch

        forward_vec_b = torch.tensor(
            [1.0, 0.0, 0.0],
            device=self.device,
        ).repeat(
            self.num_envs,
            1,
        )

        forwards = math_utils.quat_apply(
            robot_quat,
            forward_vec_b,
        )

        alignment = torch.sum(
            forwards * goal_dir,
            dim=-1,
        )

        alignment_reward = alignment * 0.5

        total_reward = (
            progress_reward
            + dist_reward
            + alignment_reward
            + backward_penalty
            + collision_reward
            + success_reward
        )

        self.extras.setdefault("log", {}).update(
            {
                "Metrics/distance": dist.mean(),
                "Metrics/velocity_to_goal": velocity_proj.mean(),
                "Metrics/alignment": alignment.mean(),
                "Metrics/current_collision_rate": collision.float().mean(),
                "Metrics/current_success_rate": reached.float().mean(),
            }
        )

        return total_reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        goal_vec = self._get_goal_vec()
        dist = torch.linalg.norm(goal_vec, dim=-1)

        reached = dist < self.cfg.target_reach_threshold
        crashed = self._get_collision()

        terminated = reached | crashed
        timeout = time_out & (~terminated)
        completed = terminated | timeout

        self.total_successes += reached.sum().item()

        self.total_collisions += (
            crashed & (~reached)
        ).sum().item()

        self.total_timeouts += timeout.sum().item()
        self.total_episodes += completed.sum().item()

        if self.total_episodes > 0:
            success_rate = (
                100.0 * self.total_successes / self.total_episodes
            )
            collision_rate = (
                100.0 * self.total_collisions / self.total_episodes
            )
            timeout_rate = (
                100.0 * self.total_timeouts / self.total_episodes
            )
        else:
            success_rate = 0.0
            collision_rate = 0.0
            timeout_rate = 0.0

        self.extras.setdefault("log", {}).update(
            {
                "Metrics/true_success_rate": success_rate,
                "Metrics/true_collision_rate": collision_rate,
                "Metrics/true_timeout_rate": timeout_rate,
                "Metrics/total_episodes": self.total_episodes,
            }
        )

        return terminated, time_out

    def _reset_idx(
        self,
        env_ids: torch.Tensor | None,
    ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        rand_yaw = torch.zeros(
            num_resets,
            3,
            device=self.device,
        )
        rand_yaw[:, 2] = (
            torch.rand(
                num_resets,
                device=self.device,
            )
            * 2.0
            * math.pi
        )

        quat = math_utils.quat_from_euler_xyz(
            rand_yaw[:, 0],
            rand_yaw[:, 1],
            rand_yaw[:, 2],
        )

        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0

        self.robot.write_root_state_to_sim(
            root_state,
            env_ids,
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(
            self.robot.data.default_joint_vel[env_ids]
        )

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )

        if self._camera_hist is not None:
            self._camera_hist[env_ids] = 0.0

        goal_radii = torch.empty(
            num_resets,
            device=self.device,
        ).uniform_(3.5, 5.0)

        goal_thetas = torch.empty(
            num_resets,
            device=self.device,
        ).uniform_(
            -math.pi,
            math.pi,
        )

        self.target_pos[env_ids, 0] = (
            self.scene.env_origins[env_ids, 0]
            + goal_radii * torch.cos(goal_thetas)
        )

        self.target_pos[env_ids, 1] = (
            self.scene.env_origins[env_ids, 1]
            + goal_radii * torch.sin(goal_thetas)
        )

        self.target_pos[env_ids, 2] = 0.25

        self.prev_dist[env_ids] = torch.linalg.norm(
            self.target_pos[env_ids] - root_state[:, :3],
            dim=-1,
        )

        for env_id in env_ids.tolist():
            origin = self.scene.env_origins[env_id, :2]
            goal = self.target_pos[env_id, :2]

            accepted_positions = []

            for obstacle_index in range(self.num_obstacles):
                while True:
                    radius = (
                        torch.rand(
                            (),
                            device=self.device,
                        )
                        * 1.5
                        + 1.5
                    )

                    theta = (
                        torch.rand(
                            (),
                            device=self.device,
                        )
                        * 2.0
                        * math.pi
                        - math.pi
                    )

                    x = origin[0] + radius * torch.cos(theta)
                    y = origin[1] + radius * torch.sin(theta)

                    candidate = torch.stack([x, y])

                    if torch.norm(candidate - origin) < 1.5:
                        continue

                    if torch.norm(candidate - goal) < 0.8:
                        continue

                    too_close = False

                    for previous in accepted_positions:
                        if torch.norm(candidate - previous) < 0.8:
                            too_close = True
                            break

                    if too_close:
                        continue

                    accepted_positions.append(candidate)

                    obstacle = self.obstacles[obstacle_index]

                    obstacle_state = (
                        obstacle.data.default_root_state[
                            env_id : env_id + 1
                        ].clone()
                    )

                    obstacle_state[0, 0] = x
                    obstacle_state[0, 1] = y
                    obstacle_state[0, 2] = (
                        self.scene.env_origins[env_id, 2]
                        + 0.20
                    )
                    obstacle_state[0, 7:] = 0.0

                    obstacle.write_root_state_to_sim(
                        obstacle_state,
                        env_ids=torch.tensor(
                            [env_id],
                            device=self.device,
                            dtype=torch.long,
                        ),
                    )

                    break

        self.actions[env_ids] = 0.0
        