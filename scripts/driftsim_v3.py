#!/usr/bin/env python
"""Generate version-3 drift-simulation data with explicit detector geometry.

V3 keeps the legacy straight-track physics and random distributions, but fixes
wire geometry, dense class ids, z coordinates, and the data schema. V1 and V2
remain untouched for reproducibility of earlier experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


SCHEMA_VERSION = 3
OUTPUT_COLUMNS = [
    "schema_version",
    "ev_id",
    "tr_id",
    "station",
    "plane",
    "wireid",
    "local_tube_id",
    "tube_class_id",
    "x",
    "y",
    "z",
    "x0",
    "y0",
    "z0",
    "dr",
    "lr",
    "vtxx",
    "vtxy",
    "pt",
    "phi",
    "theta",
    "charge",
]

PLANE_NORMALS = {
    "X": np.array([1.0, 0.0], dtype=np.float64),
    "Y": np.array([0.0, 1.0], dtype=np.float64),
    "U": np.array([1.0, 1.0], dtype=np.float64) / math.sqrt(2.0),
    "V": np.array([1.0, -1.0], dtype=np.float64) / math.sqrt(2.0),
}


@dataclass(frozen=True)
class PlaneGeometry:
    station: int
    plane_type: str
    z: float
    normal: np.ndarray
    coordinate_min: float
    coordinate_max: float
    pitch: float
    tube_count: int
    class_offset: int

    def measurement(self, x: float, y: float) -> dict[str, float | int]:
        point = np.array([x, y], dtype=np.float64)
        coordinate = float(self.normal @ point)
        scaled = (coordinate - self.coordinate_min) / self.pitch
        local_tube_id = int(math.floor(scaled + 0.5))
        if local_tube_id < 0 or local_tube_id >= self.tube_count:
            raise ValueError(
                f"Point ({x}, {y}) maps to tube {local_tube_id} outside "
                f"station {self.station} range [0, {self.tube_count})."
            )

        wire_coordinate = self.coordinate_min + local_tube_id * self.pitch
        signed_drift = coordinate - wire_coordinate
        drift = abs(signed_drift)
        lr = 1 if signed_drift >= 0 else -1
        anchor = wire_coordinate * self.normal
        return {
            "wireid": self.station * 1000 + local_tube_id,
            "local_tube_id": local_tube_id,
            "tube_class_id": self.class_offset + local_tube_id,
            "x0": float(anchor[0]),
            "y0": float(anchor[1]),
            "z0": self.z,
            "dr": drift,
            "lr": lr,
            "wire_coordinate": wire_coordinate,
        }


@dataclass(frozen=True)
class DetectorGeometry:
    half_size: float
    pitch: float
    planes: tuple[PlaneGeometry, ...]

    @property
    def total_tubes(self) -> int:
        return sum(plane.tube_count for plane in self.planes)

    def plane(self, station: int) -> PlaneGeometry:
        if station < 1 or station > len(self.planes):
            raise ValueError(f"Station {station} is outside [1, {len(self.planes)}].")
        return self.planes[station - 1]


def projection_bounds(normal: np.ndarray, half_size: float) -> tuple[float, float]:
    extent = half_size * float(np.abs(normal).sum())
    return -extent, extent


def build_detector_geometry(detector_cfg: dict) -> DetectorGeometry:
    half_size = float(detector_cfg["half_size_mm"])
    pitch = float(detector_cfg["tube_pitch_mm"])
    plane_types = [str(value).upper() for value in detector_cfg["plane_types"]]
    if not plane_types:
        raise ValueError("At least one detector plane is required.")
    if any(value not in PLANE_NORMALS for value in plane_types):
        raise ValueError(f"Unknown plane type in {plane_types}; expected X, Y, U, or V.")
    if half_size <= 0 or pitch <= 0:
        raise ValueError("Detector half-size and tube pitch must be positive.")
    z_min = float(detector_cfg["z_min_mm"])
    z_max = float(detector_cfg["z_max_mm"])
    if z_max < z_min:
        raise ValueError("detector.z_max_mm must not be less than z_min_mm.")

    z_planes = np.linspace(
        z_min,
        z_max,
        len(plane_types),
    )
    planes = []
    class_offset = 0
    for station, (plane_type, z) in enumerate(zip(plane_types, z_planes), start=1):
        normal = PLANE_NORMALS[plane_type]
        coordinate_min, coordinate_max = projection_bounds(normal, half_size)
        maximum_index = int(math.floor((coordinate_max - coordinate_min) / pitch + 0.5))
        tube_count = maximum_index + 1
        planes.append(
            PlaneGeometry(
                station=station,
                plane_type=plane_type,
                z=float(z),
                normal=normal.copy(),
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
                pitch=pitch,
                tube_count=tube_count,
                class_offset=class_offset,
            )
        )
        class_offset += tube_count

    return DetectorGeometry(half_size=half_size, pitch=pitch, planes=tuple(planes))


def line_extrapolate_to_z(
    vertex_x: float,
    vertex_y: float,
    theta: float,
    phi: float,
    z: float,
) -> tuple[float, float, float]:
    transverse = z * math.tan(theta)
    return (
        vertex_x + transverse * math.cos(phi),
        vertex_y + transverse * math.sin(phi),
        float(z),
    )


def default_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "events": 100,
        "seed": 42,
        "output_dir": "outputs/drift_sim_v3",
        "data_filename": "output.tsv",
        "config_filename": "config.yaml",
        "metadata_filename": "metadata.yaml",
        "seed_lock_filename": "seed.lock",
        "detector": {
            "half_size_mm": 750.0,
            "tube_pitch_mm": 10.0,
            "z_min_mm": 0.0,
            "z_max_mm": 240.0,
            "plane_types": ["X", "Y", "U", "V", "X", "Y", "U", "V"],
        },
        "simulation": {
            "vertex_min_mm": -700.0,
            "vertex_max_mm": 700.0,
            "tracks_per_event_min": 1,
            "tracks_per_event_max_exclusive": 10,
            "pt_min_mev": 100.0,
            "pt_max_mev": 1000.0,
            # Preserve the exact V1/V2 sampling interval (2 * 3.14156).
            "phi_max_rad": 6.28312,
            "efficiency": 1.0,
            "noise_hits_per_event": 0,
        },
    }


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML config must contain a mapping at the top level.")
    return deep_merge(default_config(), loaded)


def normalize_config(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    if args.events is not None:
        config["events"] = args.events
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.data_filename is not None:
        config["data_filename"] = args.data_filename

    config["schema_version"] = SCHEMA_VERSION
    config["events"] = int(config["events"])
    config["seed"] = int(config["seed"])
    if config["events"] < 0:
        raise ValueError("events must be greater than or equal to zero.")
    simulation = config["simulation"]
    efficiency = float(simulation["efficiency"])
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError("simulation.efficiency must be in [0, 1].")
    if float(simulation["vertex_max_mm"]) < float(simulation["vertex_min_mm"]):
        raise ValueError("simulation.vertex_max_mm must not be less than vertex_min_mm.")
    if int(simulation["tracks_per_event_min"]) < 0:
        raise ValueError("simulation.tracks_per_event_min must be non-negative.")
    if int(simulation["tracks_per_event_max_exclusive"]) <= int(
        simulation["tracks_per_event_min"]
    ):
        raise ValueError(
            "simulation.tracks_per_event_max_exclusive must exceed the minimum."
        )
    if float(simulation["pt_max_mev"]) < float(simulation["pt_min_mev"]):
        raise ValueError("simulation.pt_max_mev must not be less than pt_min_mev.")
    if float(simulation["phi_max_rad"]) <= 0.0:
        raise ValueError("simulation.phi_max_rad must be positive.")
    if int(simulation["noise_hits_per_event"]) != 0:
        raise ValueError(
            "V3 preserves the legacy no-noise physics; noise_hits_per_event must be 0."
        )
    return config


def iter_rows(config: dict, geometry: DetectorGeometry, rng: random.Random):
    simulation = config["simulation"]
    vertex_min = float(simulation["vertex_min_mm"])
    vertex_max = float(simulation["vertex_max_mm"])
    efficiency = float(simulation["efficiency"])
    min_tracks = int(simulation["tracks_per_event_min"])
    max_tracks = int(simulation["tracks_per_event_max_exclusive"])
    pt_min = float(simulation["pt_min_mev"])
    pt_max = float(simulation["pt_max_mev"])
    phi_max = float(simulation["phi_max_rad"])

    progress = tqdm(range(config["events"]), desc="Generating v3 events")
    for event_id in progress:
        vertex_x = rng.uniform(vertex_min, vertex_max)
        vertex_y = rng.uniform(vertex_min, vertex_max)
        track_count = int(rng.uniform(min_tracks, max_tracks))

        for track_id in range(track_count):
            pt = rng.uniform(pt_min, pt_max)
            phi = rng.uniform(0.0, phi_max)
            theta = math.acos(rng.uniform(0.0, 1.0))
            charge = 0

            for plane in geometry.planes:
                x, y, z = line_extrapolate_to_z(
                    vertex_x, vertex_y, theta, phi, plane.z
                )
                if abs(x) >= geometry.half_size or abs(y) >= geometry.half_size:
                    continue
                if efficiency < 1.0 and rng.random() >= efficiency:
                    continue

                measurement = plane.measurement(x, y)
                yield {
                    "schema_version": SCHEMA_VERSION,
                    "ev_id": event_id,
                    "tr_id": track_id,
                    "station": plane.station,
                    "plane": plane.plane_type,
                    "wireid": measurement["wireid"],
                    "local_tube_id": measurement["local_tube_id"],
                    "tube_class_id": measurement["tube_class_id"],
                    "x": x,
                    "y": y,
                    "z": z,
                    "x0": measurement["x0"],
                    "y0": measurement["y0"],
                    "z0": measurement["z0"],
                    "dr": measurement["dr"],
                    "lr": measurement["lr"],
                    "vtxx": vertex_x,
                    "vtxy": vertex_y,
                    "pt": pt,
                    "phi": phi,
                    "theta": theta,
                    "charge": charge,
                }


def write_data(path: Path, config: dict, geometry: DetectorGeometry) -> int:
    rng = random.Random(config["seed"])
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in iter_rows(config, geometry, rng):
            writer.writerow(row)
            row_count += 1
    return row_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(config: dict, geometry: DetectorGeometry, data_path: Path, row_count: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "columns": OUTPUT_COLUMNS,
        "events_requested": config["events"],
        "seed": config["seed"],
        "rows": row_count,
        "data_file": data_path.name,
        "data_sha256": sha256(data_path),
        "total_tubes": geometry.total_tubes,
        "detector": {
            "half_size_mm": geometry.half_size,
            "tube_pitch_mm": geometry.pitch,
            "stations": [
                {
                    "station": plane.station,
                    "plane": plane.plane_type,
                    "z_mm": plane.z,
                    "normal": plane.normal.tolist(),
                    "coordinate_min_mm": plane.coordinate_min,
                    "coordinate_max_mm": plane.coordinate_max,
                    "tube_count": plane.tube_count,
                    "class_offset": plane.class_offset,
                }
                for plane in geometry.planes
            ],
        },
    }


def save_yaml(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="configs/drift_sim_v3.yaml")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("-n", "--events", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = normalize_config(load_config(Path(args.config)), args)
    geometry = build_detector_geometry(config["detector"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / config["data_filename"]

    row_count = write_data(data_path, config, geometry)
    save_yaml(output_dir / config["config_filename"], config)
    save_yaml(
        output_dir / config["metadata_filename"],
        metadata(config, geometry, data_path, row_count),
    )
    (output_dir / config["seed_lock_filename"]).write_text(
        f"{config['seed']}\n", encoding="utf-8"
    )

    print(f"Data saved to: {data_path}")
    print(f"Rows: {row_count}")
    print(f"Total tube classes: {geometry.total_tubes}")


if __name__ == "__main__":
    main()
