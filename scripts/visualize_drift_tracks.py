#!/usr/bin/env python
"""Visualize fired straw tubes and truth trajectories from drift-sim TSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DRIFT_SIM_COLUMNS = [
    "ev_id", "wireid", "dr", "lr", "station", "tr_id",
    "x", "y", "x0", "y0", "z0",
]
Z_PLANES = np.linspace(0.0, 240.0, 8)
PLANE_TYPES = {1: "X", 2: "Y", 3: "U", 4: "V", 5: "X", 6: "Y", 7: "U", 8: "V"}
NORMALS = {
    "X": np.array([1.0, 0.0]),
    "Y": np.array([0.0, 1.0]),
    "U": np.array([1.0, 1.0]) / np.sqrt(2.0),
    "V": np.array([1.0, -1.0]) / np.sqrt(2.0),
}
DIRECTIONS = {
    "X": np.array([0.0, 1.0]),
    "Y": np.array([1.0, 0.0]),
    "U": np.array([1.0, -1.0]) / np.sqrt(2.0),
    "V": np.array([1.0, 1.0]) / np.sqrt(2.0),
}
PLANE_COLORS = {
    "X": "#1f77b4",
    "Y": "#ff7f0e",
    "U": "#2ca02c",
    "V": "#9467bd",
}


def parse_selection(value: str) -> tuple[int, int]:
    try:
        event_id, track_id = value.split(":", 1)
        return int(event_id), int(track_id)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Selection must have the form EVENT:TRACK") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--select",
        type=parse_selection,
        action="append",
        default=None,
        help="Event and track as EVENT:TRACK; repeat for multiple figures.",
    )
    parser.add_argument("--detector-half-size", type=float, default=750.0)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_tracks(
    path: Path,
    selections: list[tuple[int, int]],
    chunk_size: int,
) -> dict[tuple[int, int], pd.DataFrame]:
    wanted = set(selections)
    pieces: dict[tuple[int, int], list[pd.DataFrame]] = {key: [] for key in wanted}
    max_event = max(event_id for event_id, _ in wanted)

    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        names=DRIFT_SIM_COLUMNS,
        chunksize=chunk_size,
        engine="c",
    ):
        for event_id, track_id in wanted:
            selected = chunk[(chunk["ev_id"] == event_id) & (chunk["tr_id"] == track_id)]
            if len(selected):
                pieces[(event_id, track_id)].append(selected.copy())
        if int(chunk["ev_id"].iloc[-1]) > max_event:
            break

    result = {}
    for key, track_pieces in pieces.items():
        if not track_pieces:
            raise ValueError(f"Track {key[0]}:{key[1]} was not found in {path}")
        result[key] = pd.concat(track_pieces, ignore_index=True).sort_values("station")
    return result


def clip_line_to_square(
    normal: np.ndarray,
    coordinate: float,
    direction: np.ndarray,
    half_size: float,
) -> np.ndarray:
    """Return two endpoints of n·p=coordinate clipped to the detector square."""
    origin = coordinate * normal
    lower, upper = -np.inf, np.inf
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if abs(origin[axis]) > half_size:
                raise ValueError("Tube does not intersect the detector square")
            continue
        first = (-half_size - origin[axis]) / direction[axis]
        second = (half_size - origin[axis]) / direction[axis]
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
    if lower > upper:
        raise ValueError("Tube does not intersect the detector square")
    return np.vstack([origin + lower * direction, origin + upper * direction])


def track_geometry(track: pd.DataFrame, half_size: float) -> list[dict]:
    geometry = []
    for row in track.itertuples(index=False):
        station = int(row.station)
        plane_type = PLANE_TYPES[station]
        normal = NORMALS[plane_type]
        direction = DIRECTIONS[plane_type]
        truth_xy = np.array([float(row.x), float(row.y)])
        signed_drift = float(row.lr) * float(row.dr)
        wire_coordinate = float(normal @ truth_xy - signed_drift)
        endpoints = clip_line_to_square(normal, wire_coordinate, direction, half_size)
        closest_point = truth_xy - signed_drift * normal
        geometry.append(
            {
                "station": station,
                "plane_type": plane_type,
                "z": float(Z_PLANES[station - 1]),
                "truth_xy": truth_xy,
                "closest_point": closest_point,
                "endpoints": endpoints,
                "drift": float(row.dr),
                "lr": int(row.lr),
            }
        )
    return geometry


def draw_detector_planes_3d(ax, half_size: float) -> None:
    border = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
            [-half_size, -half_size],
        ]
    )
    for z in Z_PLANES:
        ax.plot(border[:, 0], border[:, 1], np.full(len(border), z), color="#b7bcc5", lw=0.45, alpha=0.34)


def draw_track_3d(ax, geometry: list[dict], half_size: float) -> None:
    draw_detector_planes_3d(ax, half_size)
    truth = np.array([[*item["truth_xy"], item["z"]] for item in geometry])
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], color="#d62728", lw=2.2, marker="o", ms=4.2, zorder=10)

    for item in geometry:
        endpoints = item["endpoints"]
        z = item["z"]
        color = PLANE_COLORS[item["plane_type"]]
        ax.plot(endpoints[:, 0], endpoints[:, 1], [z, z], color=color, lw=4.0, alpha=0.72)
        point = item["truth_xy"]
        closest = item["closest_point"]
        ax.plot(
            [point[0], closest[0]],
            [point[1], closest[1]],
            [z, z],
            color="#333333",
            lw=1.15,
            ls="--",
            alpha=0.9,
        )

    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_zlim(float(Z_PLANES.min()), float(Z_PLANES.max()))
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")
    ax.set_zlabel("z, mm")
    ax.set_box_aspect((1.0, 1.0, 0.52))
    ax.view_init(elev=26, azim=-58)
    ax.grid(alpha=0.2)


def draw_track_xy(ax, geometry: list[dict], half_size: float) -> None:
    truth = np.array([item["truth_xy"] for item in geometry])
    for item in geometry:
        endpoints = item["endpoints"]
        color = PLANE_COLORS[item["plane_type"]]
        ax.plot(endpoints[:, 0], endpoints[:, 1], color=color, lw=3.2, alpha=0.64)
        point = item["truth_xy"]
        closest = item["closest_point"]
        ax.plot(
            [point[0], closest[0]],
            [point[1], closest[1]],
            color="#333333",
            lw=1.0,
            ls="--",
            alpha=0.9,
        )
        ax.annotate(
            str(item["station"]),
            xy=point,
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#222222",
        )

    ax.plot(truth[:, 0], truth[:, 1], color="#d62728", lw=2.2, marker="o", ms=4.2, zorder=10)
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")
    ax.grid(alpha=0.2)


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=PLANE_COLORS[plane_type], lw=4, label=f"{plane_type} fired tube")
        for plane_type in ("X", "Y", "U", "V")
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#d62728", lw=2.2, marker="o", label="truth trajectory"),
            Line2D([0], [0], color="#333333", lw=1.15, ls="--", label="drift segment"),
        ]
    )
    return handles


def draw_pair(
    axes,
    geometry: list[dict],
    event_id: int,
    track_id: int,
    half_size: float,
) -> None:
    ax_3d, ax_xy = axes
    draw_track_3d(ax_3d, geometry, half_size)
    draw_track_xy(ax_xy, geometry, half_size)
    drift_mean = np.mean([item["drift"] for item in geometry])
    ax_3d.set_title(f"Event {event_id}, track {track_id}: 3D | {len(geometry)} hits")
    ax_xy.set_title(f"XY projection | mean drift {drift_mean:.2f} mm")


def save_individual(
    output_path: Path,
    geometry: list[dict],
    event_id: int,
    track_id: int,
    half_size: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(13.5, 6.2), constrained_layout=True)
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 2, 2)
    draw_pair((ax_3d, ax_xy), geometry, event_id, track_id, half_size)
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    output_path: Path,
    tracks: dict[tuple[int, int], list[dict]],
    selections: list[tuple[int, int]],
    half_size: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(15.5, 5.6 * len(selections)), constrained_layout=True)
    for row, selection in enumerate(selections):
        ax_3d = fig.add_subplot(len(selections), 2, row * 2 + 1, projection="3d")
        ax_xy = fig.add_subplot(len(selections), 2, row * 2 + 2)
        draw_pair((ax_3d, ax_xy), tracks[selection], selection[0], selection[1], half_size)
    fig.suptitle("Fired straw tubes and simulated truth trajectories", fontsize=15)
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selections = args.select or [(0, 0), (1, 2), (2, 3)]
    loaded = load_tracks(args.data, selections, args.chunk_size)
    geometries = {
        selection: track_geometry(loaded[selection], args.detector_half_size)
        for selection in selections
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for event_id, track_id in selections:
        output_path = args.output_dir / f"event_{event_id:06d}_track_{track_id:02d}.png"
        save_individual(
            output_path,
            geometries[(event_id, track_id)],
            event_id,
            track_id,
            args.detector_half_size,
            args.dpi,
        )

    overview_path = args.output_dir / "selected_tracks_overview.png"
    save_overview(
        overview_path,
        geometries,
        selections,
        args.detector_half_size,
        args.dpi,
    )
    print(f"Saved {len(selections)} track figures and overview to {args.output_dir}")


if __name__ == "__main__":
    main()
