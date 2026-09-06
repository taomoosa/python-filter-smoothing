"""Play a saved MPC trajectory in cuRobo's Viser robot viewer."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from curobo._src.geom.types import SceneCfg
from curobo._src.types.content_path import ContentPath
from curobo.config_io import load_yaml
from curobo.types import JointState
from curobo.viewer import ViserVisualizer

from python_filter_smoothing.continuous_trajectory import load_config

DEFAULT_MPC_CONFIG = (
    Path(__file__).parent / "python_filter_smoothing/configs/long_mpc.yml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--mpc-config",
        type=Path,
        help="override the MPC YAML recorded in summary.json",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Viser bind address; use 0.0.0.0 only on a trusted network",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--no-hold", action="store_true")
    args = parser.parse_args()
    if args.rate <= 0.0:
        raise ValueError("rate must be positive")

    summary = json.loads((args.artifact / "summary.json").read_text())
    trajectory_path = args.artifact / "trajectory.csv"
    with trajectory_path.open(newline="") as stream:
        columns = next(csv.reader(stream))
    columns[0] = columns[0].removeprefix("# ")
    values = np.loadtxt(trajectory_path, delimiter=",", skiprows=1, ndmin=2)
    column_index = {name: index for index, name in enumerate(columns)}
    joint_names = list(summary["joint_names"])
    q = np.column_stack(
        [values[:, column_index[f"q_rad_{joint}"]] for joint in joint_names]
    )

    if args.mpc_config is not None:
        mpc_config = args.mpc_config
    else:
        recorded_config = Path(summary.get("mpc_config", DEFAULT_MPC_CONFIG))
        mpc_config = (
            recorded_config
            if recorded_config.is_absolute()
            else args.artifact.resolve() / recorded_config
        )
    config = load_config(mpc_config)
    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=config["robot"]["config"]),
        connect_ip=args.host,
        connect_port=args.port,
        add_control_frames=False,
        add_robot_to_scene=True,
    )
    visualizer.add_scene(SceneCfg.create(load_yaml(config["scene"])))
    dt = float(summary["command_dt_s"]) / args.rate
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"Viser playback: http://{display_host}:{args.port}")
    try:
        for position in q:
            visualizer.set_joint_state(
                JointState.from_position(
                    torch.as_tensor(position).reshape(1, -1), joint_names=joint_names
                )
            )
            time.sleep(dt)
        while not args.no_hold:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
