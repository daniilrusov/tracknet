#!/usr/bin/env python
"""Compare drift-sim truth tracks with the corrupted features stored by the generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from visualize_drift_tracks import (
    PLANE_COLORS,
    PLANE_TYPES,
    draw_track_3d,
    legend_handles,
    load_tracks,
    parse_selection,
    track_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--select",
        type=parse_selection,
        action="append",
        default=None,
        help="Event and track as EVENT:TRACK; repeat for multiple rows.",
    )
    parser.add_argument("--detector-half-size", type=float, default=750.0)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def draw_raw_features_3d(ax, track, half_size: float) -> None:
    track = track.sort_values("station")
    raw = track[["x0", "y0", "z0"]].to_numpy(dtype=float)
    ax.plot(
        raw[:, 0],
        raw[:, 1],
        raw[:, 2],
        color="#343a40",
        lw=1.8,
        ls="--",
        alpha=0.78,
        label="stored feature sequence",
    )
    for row in track.itertuples(index=False):
        station = int(row.station)
        plane_type = PLANE_TYPES[station]
        ax.scatter(
            [row.x0],
            [row.y0],
            [row.z0],
            color=PLANE_COLORS[plane_type],
            s=42,
            depthshade=False,
            zorder=10,
        )
        ax.text(
            row.x0,
            row.y0,
            row.z0 + 7,
            f"{station}{plane_type}",
            fontsize=8,
            color="#202020",
        )

    detector_border = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
            [-half_size, -half_size],
        ]
    )
    for z in (0.0, 240.0):
        ax.plot(
            detector_border[:, 0], detector_border[:, 1], np.full(5, z),
            color="#7a8088", lw=1.0, alpha=0.72,
        )
    for corner in detector_border[:4]:
        ax.plot(
            [corner[0], corner[0]], [corner[1], corner[1]], [0.0, 240.0],
            color="#7a8088", lw=0.8, alpha=0.52,
        )

    minimum = float(min(raw[:, 0].min(), raw[:, 1].min()))
    maximum = float(max(raw[:, 0].max(), raw[:, 1].max()))
    padding = max((maximum - minimum) * 0.08, 250.0)
    ax.set_xlim(minimum - padding, maximum + padding)
    ax.set_ylim(minimum - padding, maximum + padding)
    ax.set_zlim(0.0, 240.0)
    ax.set_xlabel("stored x0, mm")
    ax.set_ylabel("stored y0, mm")
    ax.set_zlabel("stored z0, mm")
    ax.set_box_aspect((1.0, 1.0, 0.38))
    ax.view_init(elev=26, azim=-58)
    ax.grid(alpha=0.2)


def raw_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0], [0], color=PLANE_COLORS[plane_type], marker="o", lw=0,
            markersize=7, label=f"stored {plane_type} anchor",
        )
        for plane_type in ("X", "Y", "U", "V")
    ]
    handles.append(
        Line2D(
            [0], [0], color="#343a40", lw=1.8, ls="--",
            label="stored anchor sequence",
        )
    )
    handles.append(
        Line2D(
            [0], [0], color="#7a8088", lw=1.0,
            label="physical ±750 mm envelope",
        )
    )
    return handles


def draw_comparison_row(
    axes,
    track,
    event_id: int,
    track_id: int,
    half_size: float,
) -> None:
    truth_axis, raw_axis = axes
    geometry = track_geometry(track, half_size)
    draw_track_3d(truth_axis, geometry, half_size)
    draw_raw_features_3d(raw_axis, track, half_size)

    raw = track[["x0", "y0"]].to_numpy(dtype=float)
    max_anchor = float(np.abs(raw).max())
    truth_axis.set_title(f"Event {event_id}, track {track_id}: simulated truth + fired tubes")
    raw_axis.set_title(
        f"Features written by generator | max |x0/y0| = {max_anchor:,.0f} mm"
    )


def save_overview(
    path: Path,
    tracks,
    selections: list[tuple[int, int]],
    half_size: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(15.5, 5.7 * len(selections)), constrained_layout=True)
    for row, selection in enumerate(selections):
        truth_axis = fig.add_subplot(len(selections), 2, row * 2 + 1, projection="3d")
        raw_axis = fig.add_subplot(len(selections), 2, row * 2 + 2, projection="3d")
        draw_comparison_row(
            (truth_axis, raw_axis),
            tracks[selection],
            selection[0],
            selection[1],
            half_size,
        )

    fig.suptitle("Generator artifacts: physical track versus stored model features", fontsize=15)
    fig.legend(
        handles=legend_handles()[:5] + raw_legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selections = args.select or [(0, 0), (1, 2), (2, 3)]
    tracks = load_tracks(args.data, selections, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "generator_artifacts_overview.png"
    save_overview(
        output_path,
        tracks,
        selections,
        args.detector_half_size,
        args.dpi,
    )
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
