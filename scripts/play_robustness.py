import argparse
import contextlib
import importlib.metadata
import os
import queue
import threading

import torch
from isaaclab.app import AppLauncher


MODEL_NAME = "ADR Wheel + Camera + Floor"
MODEL_CHECKPOINT = (
    "/home/karthik/jetbot_rl_navigation/logs/jetbot_adr/model_2800.pt"
)

STATUS_PRINT_INTERVAL = 120
MARKER_HEIGHT_OFFSET = 0.85
MARKER_X_OFFSETS = {
    "camera": -0.30,
    "wheel": 0.00,
    "floor": 0.30,
}

GREEN = (0.1, 0.9, 0.1)
YELLOW = (1.0, 0.8, 0.0)
RED = (1.0, 0.1, 0.1)

CAMERA_TRAINING_RANGE = (0.00, 0.05)
WHEEL_TRAINING_RANGE = (0.00, 0.20)
FLOOR_STATIC_TRAINING_RANGE = (0.80, 1.20)
FLOOR_DYNAMIC_TRAINING_RANGE = (0.70, 1.10)

CRATE_BOARD_FRACTION = 0.10
CRATE_DEPTH_FRACTION = 0.08
CRATE_COLOUR = (0.45, 0.22, 0.07)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive robustness demo for the trained JetBot ADR policy."
    )

    parser.add_argument(
        "--camera_noise",
        type=float,
        default=0.0,
        help="Camera Gaussian noise standard deviation.",
    )
    parser.add_argument(
        "--wheel_friction",
        type=float,
        default=0.0,
        help="Wheel joint friction coefficient.",
    )
    parser.add_argument(
        "--floor_static",
        type=float,
        default=1.0,
        help="Floor static friction coefficient.",
    )
    parser.add_argument(
        "--floor_dynamic",
        type=float,
        default=0.9,
        help="Floor dynamic friction coefficient.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to display.",
    )

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, visualizer="kit", livestream=2)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim imports must come after AppLauncher starts the application.
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from jetbot_rl_navigation.agents.jetbot_adr_ppo_cfg import JetBotADRPPORunnerCfg
from jetbot_rl_navigation.envs.jetbot_robustness_env import JetBotRobustnessEnv
from jetbot_rl_navigation.envs.jetbot_robustness_env_cfg import JetBotRobustnessEnvCfg


command_queue = queue.Queue()
live_status_enabled = False

HOME_VALUES = {
    "camera": args_cli.camera_noise,
    "wheel": args_cli.wheel_friction,
    "floor_static": args_cli.floor_static,
    "floor_dynamic": args_cli.floor_dynamic,
}


def create_agent_cfg():
    agent_cfg = JetBotADRPPORunnerCfg()
    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    return handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)


def print_test_configuration():
    print("\n" + "=" * 65)
    print("JETBOT INTERACTIVE ROBUSTNESS TEST")
    print("=" * 65)
    print(f"Model:            {MODEL_NAME}")
    print(f"Camera noise:     {args_cli.camera_noise:.3f}")
    print(f"Wheel friction:   {args_cli.wheel_friction:.3f}")
    print(f"Floor static:     {args_cli.floor_static:.3f}")
    print(f"Floor dynamic:    {args_cli.floor_dynamic:.3f}")
    print("=" * 65 + "\n")


def print_controls():
    print("\n" + "=" * 65)
    print("LIVE CONTROLS")
    print("=" * 65)
    print("c <value>              Camera noise")
    print("w <value>              Wheel friction")
    print("f <static> <dynamic>   Floor friction")
    print("s                      Start continuous live status")
    print("g                      Stop continuous live status")
    print("r                      Reset to starting values")
    print("h                      Show controls")

    print("\nADR TRAINING RANGES")
    print("-" * 65)
    print(f"Camera noise:     {CAMERA_TRAINING_RANGE[0]:.2f} - {CAMERA_TRAINING_RANGE[1]:.2f}")
    print(f"Wheel friction:   {WHEEL_TRAINING_RANGE[0]:.2f} - {WHEEL_TRAINING_RANGE[1]:.2f}")
    print(
        f"Floor static:     {FLOOR_STATIC_TRAINING_RANGE[0]:.2f} - "
        f"{FLOOR_STATIC_TRAINING_RANGE[1]:.2f}"
    )
    print(
        f"Floor dynamic:    {FLOOR_DYNAMIC_TRAINING_RANGE[0]:.2f} - "
        f"{FLOOR_DYNAMIC_TRAINING_RANGE[1]:.2f}"
    )
    print("Values outside these ranges can be used for stronger robustness tests.")

    print("\nVISUAL MARKERS")
    print("-" * 65)
    print("Left   = Camera noise")
    print("Centre = Wheel friction")
    print("Right  = Floor friction")
    print("Green  = Low / near normal")
    print("Yellow = Within higher ADR training range")
    print("Red    = Outside ADR training range")
    print("=" * 65 + "\n")


def terminal_input_loop():
    while True:
        try:
            command = input().strip()
            if command:
                command_queue.put(command)
        except (EOFError, KeyboardInterrupt):
            break


def create_marker(stage, path, radius=0.10):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*GREEN)])
    return {
        "sphere": sphere,
        "translate_op": sphere.AddTranslateOp(),
    }


def create_status_markers():
    stage = omni.usd.get_context().get_stage()
    return {
        "camera": create_marker(stage, "/World/RobustnessStatus/Camera"),
        "wheel": create_marker(stage, "/World/RobustnessStatus/Wheel"),
        "floor": create_marker(stage, "/World/RobustnessStatus/Floor"),
    }


def _set_cube_visual(stage, path, center, size, colour):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))

    if cube.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        collision_api = UsdPhysics.CollisionAPI(cube.GetPrim())
        collision_api.GetCollisionEnabledAttr().Set(False)


def _hide_obstacle_visual_geometry(obstacle_prim):
    for prim in Usd.PrimRange(obstacle_prim):
        if prim == obstacle_prim:
            continue
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(prim).MakeInvisible()


def _find_obstacle_geometry(obstacle_prim):
    fallback = None

    for prim in Usd.PrimRange(obstacle_prim):
        if prim == obstacle_prim or not prim.IsA(UsdGeom.Gprim):
            continue

        if fallback is None:
            fallback = prim

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI(prim)
            enabled_attr = collision_api.GetCollisionEnabledAttr()
            if not enabled_attr or enabled_attr.Get() is not False:
                return prim

    return fallback


def _create_crate_visual(stage, root_path, minimum, maximum, local_transform):
    min_x, min_y, min_z = minimum
    max_x, max_y, max_z = maximum

    width = max(max_x - min_x, 0.05)
    depth = max(max_y - min_y, 0.05)
    height = max(max_z - min_z, 0.05)

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    cz = (min_z + max_z) * 0.5

    board = max(min(width, depth, height) * CRATE_BOARD_FRACTION, 0.02)
    face_depth = max(min(width, depth) * CRATE_DEPTH_FRACTION, 0.015)

    x_edge = width * 0.5 - board * 0.5
    y_edge = depth * 0.5 - board * 0.5
    post_height = max(height - 2.0 * board, board)

    pieces = []

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            pieces.append(
                ((cx + sx * x_edge, cy + sy * y_edge, cz),
                 (board, board, post_height))
            )

    for sz in (-1.0, 1.0):
        z = cz + sz * (height * 0.5 - board * 0.5)
        for sy in (-1.0, 1.0):
            pieces.append(
                ((cx, cy + sy * y_edge, z),
                 (width, board, board))
            )

    for sz in (-1.0, 1.0):
        z = cz + sz * (height * 0.5 - board * 0.5)
        for sx in (-1.0, 1.0):
            pieces.append(
                ((cx + sx * x_edge, cy, z),
                 (board, depth, board))
            )

    for sy in (-1.0, 1.0):
        y = cy + sy * (depth * 0.5 - face_depth * 0.5)
        for frac in (-0.28, 0.0, 0.28):
            pieces.append(
                ((cx, y, cz + frac * height),
                 (max(width - 2.0 * board, board), face_depth, board))
            )

    for sx in (-1.0, 1.0):
        x = cx + sx * (width * 0.5 - face_depth * 0.5)
        for frac in (-0.28, 0.0, 0.28):
            pieces.append(
                ((x, cy, cz + frac * height),
                 (face_depth, max(depth - 2.0 * board, board), board))
            )

    root = UsdGeom.Xform.Define(stage, root_path)
    root_xform = UsdGeom.Xformable(root.GetPrim())
    root_xform.MakeMatrixXform().Set(local_transform)

    for index, (center, size) in enumerate(pieces):
        _set_cube_visual(
            stage,
            root_path.AppendChild(f"Board_{index:02d}"),
            center,
            size,
            CRATE_COLOUR,
        )

    return root


def replace_obstacle_visuals_with_crates():
    stage = omni.usd.get_context().get_stage()
    obstacle_prims = []

    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if not name.startswith("obstacle"):
            continue

        parent = prim.GetParent()
        parent_name = parent.GetName().lower() if parent else ""
        if parent_name.startswith("obstacle"):
            continue

        if prim.IsA(UsdGeom.Xformable):
            obstacle_prims.append(prim)

    if not obstacle_prims:
        print("[WARNING] No obstacle prims were found for crate replacement.")
        return

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    created = 0

    for index, obstacle_prim in enumerate(obstacle_prims):
        geometry_prim = _find_obstacle_geometry(obstacle_prim)
        if geometry_prim is None:
            print(f"[WARNING] No geometry found under {obstacle_prim.GetPath()}.")
            continue

        local_box = bbox_cache.ComputeLocalBound(geometry_prim).ComputeAlignedBox()
        minimum = local_box.GetMin()
        maximum = local_box.GetMax()

        geometry_xform = UsdGeom.Xformable(geometry_prim)
        local_transform = geometry_xform.GetLocalTransformation()

        _hide_obstacle_visual_geometry(obstacle_prim)

        crate_path = obstacle_prim.GetPath().AppendChild(f"DemoCrate_{index:02d}")
        _create_crate_visual(
            stage,
            crate_path,
            minimum,
            maximum,
            local_transform,
        )
        created += 1


def set_marker_colour(marker, colour):
    marker["sphere"].GetDisplayColorAttr().Set([Gf.Vec3f(*colour)])


def get_camera_colour(value):
    if value <= 0.02:
        return GREEN
    if value <= CAMERA_TRAINING_RANGE[1]:
        return YELLOW
    return RED


def get_wheel_colour(value):
    if value <= 0.08:
        return GREEN
    if value <= WHEEL_TRAINING_RANGE[1]:
        return YELLOW
    return RED


def get_floor_colour(static_value, dynamic_value):
    near_nominal = (
        0.95 <= static_value <= 1.05
        and 0.85 <= dynamic_value <= 0.95
    )
    if near_nominal:
        return GREEN

    inside_training_range = (
        FLOOR_STATIC_TRAINING_RANGE[0]
        <= static_value
        <= FLOOR_STATIC_TRAINING_RANGE[1]
        and FLOOR_DYNAMIC_TRAINING_RANGE[0]
        <= dynamic_value
        <= FLOOR_DYNAMIC_TRAINING_RANGE[1]
    )
    return YELLOW if inside_training_range else RED


def update_marker_colours(base_env, markers):
    cfg = base_env.cfg

    set_marker_colour(
        markers["camera"],
        get_camera_colour(cfg.robustness_camera_noise_std),
    )
    set_marker_colour(
        markers["wheel"],
        get_wheel_colour(cfg.robustness_wheel_friction),
    )
    set_marker_colour(
        markers["floor"],
        get_floor_colour(
            cfg.robustness_floor_static_friction,
            cfg.robustness_floor_dynamic_friction,
        ),
    )


def update_marker_positions(base_env, markers):
    root_position = base_env.robot.data.root_pos_w
    if hasattr(root_position, "torch"):
        root_position = root_position.torch

    x, y, z = root_position[0, :3].tolist()
    marker_height = z + MARKER_HEIGHT_OFFSET

    for name, x_offset in MARKER_X_OFFSETS.items():
        markers[name]["translate_op"].Set(
            Gf.Vec3d(x + x_offset, y, marker_height)
        )


def print_live_status(base_env):
    root_velocity = base_env.robot.data.root_lin_vel_w
    if hasattr(root_velocity, "torch"):
        root_velocity = root_velocity.torch

    velocity = torch.linalg.norm(root_velocity[0, :2]).item()
    goal_distance = torch.linalg.norm(base_env._get_goal_vec()[0]).item()
    cfg = base_env.cfg

    print("\n" + "-" * 55)
    print("LIVE ROBOT STATUS")
    print("-" * 55)
    print(f"Camera noise:      {cfg.robustness_camera_noise_std:.3f}")
    print(f"Wheel friction:    {cfg.robustness_wheel_friction:.3f}")
    print(f"Floor static:      {cfg.robustness_floor_static_friction:.3f}")
    print(f"Floor dynamic:     {cfg.robustness_floor_dynamic_friction:.3f}")
    print()
    print(f"Robot speed:       {velocity:.3f} m/s")
    print(f"Distance to goal:  {goal_distance:.3f} m")
    print("-" * 55)


def apply_camera_noise(base_env, value):
    base_env.cfg.robustness_camera_noise_std = value


def apply_wheel_friction(base_env, value):
    base_env.cfg.robustness_wheel_friction = value
    base_env._apply_test_wheel_friction(base_env.robot._ALL_INDICES)


def apply_floor_friction(base_env, static_value, dynamic_value):
    base_env.cfg.robustness_floor_static_friction = static_value
    base_env.cfg.robustness_floor_dynamic_friction = dynamic_value
    base_env._apply_test_floor_friction()


def reset_test_conditions(base_env, markers):
    apply_camera_noise(base_env, HOME_VALUES["camera"])
    apply_wheel_friction(base_env, HOME_VALUES["wheel"])
    apply_floor_friction(
        base_env,
        HOME_VALUES["floor_static"],
        HOME_VALUES["floor_dynamic"],
    )
    update_marker_colours(base_env, markers)

    print("\n[RESET] Starting test conditions restored")
    print(f"Camera noise:     {HOME_VALUES['camera']:.3f}")
    print(f"Wheel friction:   {HOME_VALUES['wheel']:.3f}")
    print(f"Floor static:     {HOME_VALUES['floor_static']:.3f}")
    print(f"Floor dynamic:    {HOME_VALUES['floor_dynamic']:.3f}")


def process_command(command, base_env, markers):
    global live_status_enabled

    parts = command.lower().split()
    if not parts:
        return

    command_name = parts[0]

    try:
        if command_name == "c":
            if len(parts) != 2:
                print("Usage: c <value>")
                return

            value = float(parts[1])
            if value < 0.0:
                print("Camera noise must be >= 0.")
                return

            apply_camera_noise(base_env, value)
            update_marker_colours(base_env, markers)
            print(f"\n[UPDATED] Camera noise = {value:.3f}")

        elif command_name == "w":
            if len(parts) != 2:
                print("Usage: w <value>")
                return

            value = float(parts[1])
            if value < 0.0:
                print("Wheel friction must be >= 0.")
                return

            apply_wheel_friction(base_env, value)
            update_marker_colours(base_env, markers)
            print(f"\n[UPDATED] Wheel friction = {value:.3f}")

        elif command_name == "f":
            if len(parts) != 3:
                print("Usage: f <static> <dynamic>")
                return

            static_value = float(parts[1])
            dynamic_value = float(parts[2])
            if static_value < 0.0 or dynamic_value < 0.0:
                print("Floor friction values must be >= 0.")
                return

            apply_floor_friction(base_env, static_value, dynamic_value)
            update_marker_colours(base_env, markers)
            print(
                "\n[UPDATED] Floor friction = "
                f"{static_value:.3f} / {dynamic_value:.3f}"
            )

        elif command_name == "s":
            live_status_enabled = True
            print("\n[LIVE STATUS] Started")
            print_live_status(base_env)

        elif command_name == "g":
            live_status_enabled = False
            print("\n[LIVE STATUS] Stopped")

        elif command_name == "r":
            reset_test_conditions(base_env, markers)

        elif command_name == "h":
            print_controls()

        else:
            print(f"\nUnknown command: {command_name}")
            print("Use c, w, f, s, g, r or h.")

    except ValueError:
        print("\nInvalid value. Please enter a number.")
    except Exception as error:
        print(f"\n[ERROR] Could not apply command: {error}")


def create_environment():
    env_cfg = JetBotRobustnessEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.robustness_camera_noise_std = args_cli.camera_noise
    env_cfg.robustness_wheel_friction = args_cli.wheel_friction
    env_cfg.robustness_floor_static_friction = args_cli.floor_static
    env_cfg.robustness_floor_dynamic_friction = args_cli.floor_dynamic

    env = JetBotRobustnessEnv(cfg=env_cfg)
    replace_obstacle_visuals_with_crates()
    return RslRlVecEnvWrapper(env)


def load_policy(env, agent_cfg):
    base_env = env.unwrapped

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=base_env.device,
        )
        runner.load(MODEL_CHECKPOINT)
        policy = runner.get_inference_policy(device=base_env.device)

    return policy


def main():
    print_test_configuration()

    agent_cfg = create_agent_cfg()
    env = create_environment()
    base_env = env.unwrapped
    policy = load_policy(env, agent_cfg)
    obs = env.get_observations()

    markers = create_status_markers()
    update_marker_colours(base_env, markers)
    update_marker_positions(base_env, markers)

    print_controls()

    input_thread = threading.Thread(target=terminal_input_loop, daemon=True)
    input_thread.start()

    step_count = 0

    while simulation_app.is_running():
        while not command_queue.empty():
            process_command(command_queue.get(), base_env, markers)

        with torch.inference_mode():
            actions = policy(obs)

        obs, _, _, _ = env.step(actions)
        update_marker_positions(base_env, markers)
        step_count += 1

        if live_status_enabled and step_count % STATUS_PRINT_INTERVAL == 0:
            print_live_status(base_env)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
