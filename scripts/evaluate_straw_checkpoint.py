#!/usr/bin/env python
"""Evaluate a straw TrackNET checkpoint on a fixed drift-sim TSV dataset.

The report contains micro top-k recall, cross-entropy, MRR, track-level recall,
and breakdowns by prediction step, target station, station transition, track
length, and target tube class. Scoring starts after the number of seed hits
stored in the checkpoint (or after one seed hit for legacy checkpoints).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import types
from collections.abc import Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracknet.model import StrawTrackNET
from src.tracknet.straw_geometry import tube_local_targets, reconstruct_tube_position


DRIFT_SIM_COLUMNS = [
    "ev_id", "wireid", "dr", "lr", "station", "tr_id",
    "x", "y", "x0", "y0", "z0",
]
LEGACY_INPUT_COLUMNS = ["x0", "y0", "z0", "dr", "lr", "station"]
GEOMETRY_INPUT_COLUMNS = ["x0", "y0", "z0", "dr", "station"]
TOP_K = (1, 3, 5, 10)


def to_jsonable(value):
    """Convert checkpoint metadata containers to plain JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


@dataclass
class TrackRecord:
    inputs: np.ndarray
    targets: np.ndarray
    source_stations: np.ndarray
    target_stations: np.ndarray
    coords_current: np.ndarray | None = None
    coords_next: np.ndarray | None = None
    measurements_next: np.ndarray | None = None


@dataclass
class CoordinateStats:
    support: int = 0
    squared_error: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    absolute_error: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance_sum: float = 0.0

    def update(self, prediction, target):
        error = np.asarray(prediction, dtype=np.float64) - target
        if not np.isfinite(error).all():
            raise ValueError("Non-finite coordinate predictions or targets during evaluation.")
        self.support += len(error)
        self.squared_error += np.square(error).sum(axis=0)
        self.absolute_error += np.abs(error).sum(axis=0)
        self.distance_sum += float(np.linalg.norm(error, axis=-1).sum())

    def as_dict(self):
        return {
            "support": self.support,
            "rmse_distance_mm": float(np.sqrt(self.squared_error.sum() / self.support)),
            "mean_distance_mm": self.distance_sum / self.support,
            "rmse_xyz_mm": np.sqrt(self.squared_error / self.support).tolist(),
            "mae_xyz_mm": (self.absolute_error / self.support).tolist(),
        }


@dataclass
class TubeLocalStats:
    support: int = 0
    along_squared_error: float = 0.
    along_absolute_error: float = 0.
    lr_correct: int = 0
    lr_bce_sum: float = 0.
    hard_xyz: CoordinateStats = field(default_factory=CoordinateStats)
    soft_xyz: CoordinateStats = field(default_factory=CoordinateStats)

    def update(self, along, lr_logits, truth, measurement):
        if len(truth) == 0:
            return
        target_along, target_lr, _ = tube_local_targets(truth, measurement[..., :3], measurement[..., 4])
        if not all(torch.isfinite(value).all() for value in (along, lr_logits, truth)):
            raise ValueError("Non-finite tube-local predictions or truth during evaluation.")
        error = along.double() - target_along.double()
        self.support += len(truth)
        self.along_squared_error += float(error.square().sum())
        self.along_absolute_error += float(error.abs().sum())
        self.lr_correct += int(((lr_logits >= 0) == target_lr.bool()).sum())
        self.lr_bce_sum += float(F.binary_cross_entropy_with_logits(lr_logits.float(), target_lr, reduction='sum'))
        for soft, stats in ((False, self.hard_xyz), (True, self.soft_xyz)):
            xyz = reconstruct_tube_position(along, lr_logits, measurement, soft=soft)
            stats.update(xyz.cpu().numpy(), truth.cpu().numpy())

    def as_dict(self):
        if not self.support:
            return {"support": 0}
        return {
            "support": self.support,
            "along_rmse_mm": math.sqrt(self.along_squared_error / self.support),
            "along_mae_mm": self.along_absolute_error / self.support,
            "lr_accuracy": self.lr_correct / self.support,
            "lr_binary_cross_entropy": self.lr_bce_sum / self.support,
            "xyz_hard_given_measurement": self.hard_xyz.as_dict(),
            "xyz_soft_given_measurement": self.soft_xyz.as_dict(),
        }


@dataclass(frozen=True)
class DatasetSchema:
    version: str
    num_tubes: int
    station_offsets: np.ndarray
    station_tube_counts: np.ndarray
    metadata_path: Path | None = None
    data_sha256: str | None = None
    data_filename: str | None = None

    @property
    def num_stations(self) -> int:
        return len(self.station_offsets)

    @property
    def class_to_station(self) -> np.ndarray:
        return np.concatenate(
            [
                np.full(count, station, dtype=np.int64)
                for station, count in enumerate(self.station_tube_counts, start=1)
            ]
        )

    @property
    def class_to_local_tube(self) -> np.ndarray:
        return np.concatenate(
            [np.arange(count, dtype=np.int64) for count in self.station_tube_counts]
        )


@dataclass
class MetricStats:
    support: int = 0
    nll_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    topk_hits: dict[int, int] = field(
        default_factory=lambda: {k: 0 for k in TOP_K}
    )

    def update(
        self,
        nll: np.ndarray,
        ranks: np.ndarray,
        topk_correct: dict[int, np.ndarray],
        mask: np.ndarray | None = None,
    ) -> None:
        if mask is None:
            mask = np.ones(len(nll), dtype=bool)
        count = int(mask.sum())
        if count == 0:
            return
        self.support += count
        self.nll_sum += float(nll[mask].sum())
        self.reciprocal_rank_sum += float((1.0 / ranks[mask]).sum())
        for k in TOP_K:
            self.topk_hits[k] += int(topk_correct[k][mask].sum())

    def as_dict(self) -> dict[str, float | int]:
        if self.support == 0:
            return {
                "support": 0,
                "cross_entropy": float("nan"),
                "perplexity": float("nan"),
                "mrr": float("nan"),
                **{f"top{k}_recall": float("nan") for k in TOP_K},
            }
        cross_entropy = self.nll_sum / self.support
        return {
            "support": self.support,
            "cross_entropy": cross_entropy,
            "perplexity": math.exp(cross_entropy),
            "mrr": self.reciprocal_rank_sum / self.support,
            **{
                f"top{k}_recall": self.topk_hits[k] / self.support
                for k in TOP_K
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-hits", type=int, default=8)
    parser.add_argument(
        "--seed-hits",
        type=int,
        help="Known hits before scoring starts (default: checkpoint seed_hits or 1).",
    )
    parser.add_argument(
        "--schema-version",
        choices=["auto", "legacy", "v3"],
        default="auto",
        help="Dataset schema. Auto detects a V3 header and otherwise uses legacy.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="V3 metadata.yaml (default: next to the TSV file).",
    )
    parser.add_argument("--num-stations", type=int, default=8)
    parser.add_argument("--tubes-per-station", type=int, default=151)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: CUDA when available).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_schema(args: argparse.Namespace) -> DatasetSchema:
    with args.data.open("r", encoding="utf-8") as stream:
        first_line = stream.readline()
    detected_v3 = first_line.split("\t", 1)[0].strip() == "schema_version"
    version = args.schema_version
    if version == "auto":
        version = "v3" if detected_v3 else "legacy"

    if version == "legacy":
        if detected_v3:
            raise ValueError("V3 header found but --schema-version=legacy was requested.")
        counts = np.full(args.num_stations, args.tubes_per_station, dtype=np.int64)
        offsets = np.arange(args.num_stations, dtype=np.int64) * args.tubes_per_station
        return DatasetSchema(
            version="legacy",
            num_tubes=int(counts.sum()),
            station_offsets=offsets,
            station_tube_counts=counts,
        )

    if not detected_v3:
        raise ValueError("V3 dataset must contain a schema_version header.")
    metadata_path = args.metadata or args.data.parent / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"V3 metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    if int(metadata.get("schema_version", -1)) != 3:
        raise ValueError(f"Expected schema_version=3 in {metadata_path}.")
    stations = metadata.get("detector", {}).get("stations", [])
    if not stations:
        raise ValueError(f"Station geometry is missing from {metadata_path}.")
    station_numbers = [int(station["station"]) for station in stations]
    if station_numbers != list(range(1, len(stations) + 1)):
        raise ValueError(f"V3 stations must be consecutive; got {station_numbers}.")
    counts = np.asarray([int(station["tube_count"]) for station in stations], dtype=np.int64)
    offsets = np.asarray([int(station["class_offset"]) for station in stations], dtype=np.int64)
    expected_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(counts[:-1]))
    )
    if np.any(counts <= 0) or not np.array_equal(offsets, expected_offsets):
        raise ValueError("V3 metadata does not define contiguous positive class ranges.")
    num_tubes = int(metadata["total_tubes"])
    if int(counts.sum()) != num_tubes:
        raise ValueError("V3 station tube counts do not sum to total_tubes.")
    return DatasetSchema(
        version="v3",
        num_tubes=num_tubes,
        station_offsets=offsets,
        station_tube_counts=counts,
        metadata_path=metadata_path,
        data_sha256=metadata.get("data_sha256"),
        data_filename=metadata.get("data_file"),
    )


def iter_complete_events(
    path: Path,
    chunk_size: int,
    schema_version: str,
) -> Iterator[tuple[int, pd.DataFrame]]:
    carry: pd.DataFrame | None = None
    read_kwargs = {
        "filepath_or_buffer": path,
        "sep": r"\s+",
        "engine": "c",
        "chunksize": chunk_size,
    }
    if schema_version == "legacy":
        read_kwargs["names"] = DRIFT_SIM_COLUMNS
    else:
        read_kwargs["header"] = 0
    reader = pd.read_csv(**read_kwargs)
    try:
        for chunk in reader:
            if carry is not None and len(carry):
                chunk = pd.concat([carry, chunk], ignore_index=True)

            last_event_id = int(chunk["ev_id"].iloc[-1])
            complete = chunk[chunk["ev_id"] != last_event_id]
            carry = chunk[chunk["ev_id"] == last_event_id]
            for event_id, event in complete.groupby("ev_id", sort=False):
                yield int(event_id), event

        if carry is not None and len(carry):
            for event_id, event in carry.groupby("ev_id", sort=False):
                yield int(event_id), event
    finally:
        reader.close()


def tube_class_ids(group: pd.DataFrame, schema: DatasetSchema) -> np.ndarray:
    if schema.version == "v3":
        class_ids = group["tube_class_id"].to_numpy(dtype=np.int64)
        stations = group["station"].to_numpy(dtype=np.int64)
        local_ids = group["local_tube_id"].to_numpy(dtype=np.int64)
        if np.any(stations < 1) or np.any(stations > schema.num_stations):
            raise ValueError("V3 data contains a station outside metadata geometry.")
        station_indexes = stations - 1
        if np.any(local_ids < 0) or np.any(
            local_ids >= schema.station_tube_counts[station_indexes]
        ):
            raise ValueError("V3 data contains an out-of-range local_tube_id.")
        expected = schema.station_offsets[station_indexes] + local_ids
        if not np.array_equal(class_ids, expected):
            raise ValueError("V3 tube_class_id is inconsistent with station geometry.")
        return class_ids

    stations = group["station"].to_numpy(dtype=np.int64)
    if np.any(stations < 1) or np.any(stations > schema.num_stations):
        raise ValueError("Legacy data contains a station outside configured geometry.")
    local_tubes = group["wireid"].to_numpy(dtype=np.int64) % 1000
    local_tubes %= int(schema.station_tube_counts[0])
    return schema.station_offsets[stations - 1] + local_tubes


def iter_tracks(
    path: Path,
    chunk_size: int,
    min_hits: int,
    max_hits: int,
    schema: DatasetSchema,
    input_columns: list[str],
    include_coordinates: bool = False,
) -> Iterator[TrackRecord]:
    for _, event in iter_complete_events(path, chunk_size, schema.version):
        for _, group in event.groupby("tr_id", sort=False):
            group = group.sort_values("station")
            if not min_hits <= len(group) <= max_hits:
                continue
            hits = group[input_columns].to_numpy(dtype=np.float32, copy=True)
            stations = group["station"].to_numpy(dtype=np.int64, copy=True)
            tube_ids = tube_class_ids(group, schema)
            coordinates = None
            if include_coordinates:
                coordinates = group[["x", "y", "z" if "z" in group.columns else "z0"]].to_numpy(dtype=np.float32, copy=True)
            yield TrackRecord(
                inputs=hits[:-1],
                targets=tube_ids[1:],
                source_stations=stations[:-1],
                target_stations=stations[1:],
                coords_current=coordinates[:-1] if coordinates is not None else None,
                coords_next=coordinates[1:] if coordinates is not None else None,
                measurements_next=group[GEOMETRY_INPUT_COLUMNS].to_numpy(dtype=np.float32)[1:]
                if include_coordinates else None,
            )


def batched(records: Iterable[TrackRecord], batch_size: int) -> Iterator[list[TrackRecord]]:
    batch: list[TrackRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[StrawTrackNET, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except ModuleNotFoundError as error:
        if error.name != "lightning_fabric":
            raise

        # Lightning stores its dict-like hyperparameter wrapper by import path.
        # Evaluation only needs it to behave as a plain dict, so provide a narrow
        # compatibility shim when Lightning is not installed in the inference env.
        fabric_module = types.ModuleType("lightning_fabric")
        utilities_module = types.ModuleType("lightning_fabric.utilities")
        data_module = types.ModuleType("lightning_fabric.utilities.data")
        attribute_dict = type(
            "AttributeDict",
            (dict,),
            {"__module__": "lightning_fabric.utilities.data"},
        )
        data_module.AttributeDict = attribute_dict
        fabric_module.utilities = utilities_module
        utilities_module.data = data_module
        sys.modules["lightning_fabric"] = fabric_module
        sys.modules["lightning_fabric.utilities"] = utilities_module
        sys.modules["lightning_fabric.utilities.data"] = data_module
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", checkpoint.get("hyperparameters", {}))
    use_plane_embedding = bool(hparams.get("use_plane_embedding", False))
    model = StrawTrackNET(
        input_features=int(hparams.get("input_features", 6)),
        hidden_features=int(hparams.get("hidden_features", 128)),
        num_tubes=int(hparams.get("num_tubes", 1208)),
        batch_first=bool(hparams.get("batch_first", True)),
        use_plane_embedding=use_plane_embedding,
        plane_embedding_dim=int(hparams.get("plane_embedding_dim", 8)),
        continuous_feature_center=hparams.get(
            "continuous_feature_center", (0.0, 0.0, 120.0, 2.5)
        ),
        continuous_feature_scale=hparams.get(
            "continuous_feature_scale", (750.0, 750.0, 120.0, 2.5)
        ),
        predict_coordinates=bool(hparams.get("predict_coordinates", False)),
        coordinate_center=hparams.get("coordinate_center", (0.0, 0.0, 120.0)),
        coordinate_scale=hparams.get("coordinate_scale", (750.0, 750.0, 120.0)),
        coordinate_representation=hparams.get("coordinate_representation", "cartesian"),
        along_coordinate_scale=float(hparams.get("along_coordinate_scale", 750.0)),
    )
    model_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    checkpoint_info = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "hyper_parameters": to_jsonable(hparams),
    }
    return model, checkpoint_info


def make_batch(
    records: list[TrackRecord],
    max_len: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(records)
    input_features = records[0].inputs.shape[1]
    inputs = np.zeros((batch_size, max_len, input_features), dtype=np.float32)
    targets = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    source_stations = np.zeros((batch_size, max_len), dtype=np.int64)
    target_stations = np.zeros((batch_size, max_len), dtype=np.int64)
    lengths = np.zeros(batch_size, dtype=np.int64)
    for index, record in enumerate(records):
        length = len(record.targets)
        inputs[index, :length] = record.inputs
        targets[index, :length] = record.targets
        mask[index, :length] = True
        source_stations[index, :length] = record.source_stations
        target_stations[index, :length] = record.target_stations
        lengths[index] = length
    return (
        torch.from_numpy(inputs),
        targets,
        mask,
        source_stations,
        target_stations,
        lengths,
    )


def group_rows(grouped_stats: dict[tuple[str, str], MetricStats]) -> list[dict]:
    rows = []
    for (dimension, value), stats in sorted(grouped_stats.items()):
        rows.append({"dimension": dimension, "value": value, **stats.as_dict()})
    return rows


def station_masked_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_stations: torch.Tensor,
    schema: DatasetSchema,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Metrics within the known target station's physical class range."""
    nll = torch.empty(len(targets), dtype=logits.dtype, device=logits.device)
    ranks = torch.empty(len(targets), dtype=torch.int64, device=logits.device)
    topk_correct = {
        k: torch.empty(len(targets), dtype=torch.bool, device=logits.device)
        for k in TOP_K
    }
    for station in torch.unique(target_stations).tolist():
        station = int(station)
        if station < 1 or station > schema.num_stations:
            raise ValueError(f"Target station {station} is outside dataset geometry.")
        rows = target_stations == station
        start = int(schema.station_offsets[station - 1])
        count = int(schema.station_tube_counts[station - 1])
        station_logits = logits[rows, start:start + count]
        local_targets = targets[rows] - start
        nll[rows] = F.cross_entropy(station_logits, local_targets, reduction="none")
        true_logits = station_logits.gather(1, local_targets[:, None]).squeeze(1)
        ranks[rows] = (station_logits > true_logits[:, None]).sum(dim=1) + 1
        top10 = station_logits.topk(min(max(TOP_K), count), dim=1).indices
        for k in TOP_K:
            topk_correct[k][rows] = (
                top10[:, :min(k, count)] == local_targets[:, None]
            ).any(dim=1)
    return nll, ranks, topk_correct


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model, checkpoint_info = load_model(args.checkpoint, device)
    checkpoint_seed_hits = int(
        checkpoint_info.get("hyper_parameters", {}).get("seed_hits", 1)
    )
    seed_hits = checkpoint_seed_hits if args.seed_hits is None else int(args.seed_hits)
    if seed_hits < 1:
        raise ValueError("seed_hits must be at least 1.")
    schema = load_dataset_schema(args)
    if schema.data_filename is not None and schema.data_filename != args.data.name:
        raise ValueError(
            f"V3 metadata describes {schema.data_filename}, not {args.data.name}."
        )
    dataset_sha256 = sha256(args.data)
    if schema.data_sha256 is not None and schema.data_sha256 != dataset_sha256:
        raise ValueError("V3 dataset SHA-256 does not match metadata.yaml.")
    if model.num_tubes != schema.num_tubes:
        raise ValueError(
            f"Checkpoint has {model.num_tubes} output classes, but the "
            f"{schema.version} dataset defines {schema.num_tubes}."
        )
    relative = model.predict_coordinates and model.coordinate_representation == 'tube_relative'
    if relative and schema.version != 'v3':
        raise ValueError("Tube-relative heads require DriftSim V3 evaluation data.")
    max_len = args.max_hits - 1
    warmup_steps = seed_hits - 1
    if max_len <= warmup_steps:
        raise ValueError(
            f"max_hits={args.max_hits} leaves no targets after {seed_hits} seed hits."
        )
    num_tubes = model.num_tubes
    input_columns = (
        GEOMETRY_INPUT_COLUMNS
        if model.use_plane_embedding
        else LEGACY_INPUT_COLUMNS
    )
    class_to_station = torch.from_numpy(schema.class_to_station).to(device)
    class_to_station_np = schema.class_to_station
    class_to_local_np = schema.class_to_local_tube

    overall = MetricStats()
    station_masked_overall = MetricStats()
    grouped_stats: dict[tuple[str, str], MetricStats] = defaultdict(MetricStats)
    tube_stats = [MetricStats() for _ in range(num_tubes)]
    tracks = 0
    track_top1_recall_sum = 0.0
    track_exact = {k: 0 for k in TOP_K}
    station_correct = 0
    coordinate_stats = {name: CoordinateStats() for name in ("current", "next")}
    relative_stats = {name: TubeLocalStats() for name in ('current', 'next_relative_to_true_tube', 'next_when_tube_correct')}
    relative_station_stats = defaultdict(TubeLocalStats)

    records = iter_tracks(
        args.data,
        args.chunk_size,
        args.min_hits,
        args.max_hits,
        schema,
        input_columns,
        include_coordinates=model.predict_coordinates,
    )
    progress = tqdm(batched(records, args.batch_size), desc="Evaluating", unit="batch")
    for batch_records in progress:
        inputs, targets, mask, source_stations, target_stations, lengths = make_batch(
            batch_records, max_len
        )
        target_lengths = lengths.copy()
        supervised_lengths = target_lengths - warmup_steps
        if np.any(supervised_lengths <= 0):
            raise ValueError(
                f"Every evaluated track must contain more than {seed_hits} hits."
            )
        mask[:, :warmup_steps] = False
        # Padding is only appended after each valid prefix, so running the GRU on
        # the fixed seven-step tensor gives identical valid-step outputs and avoids
        # pack/unpack overhead during this large, inference-only pass.
        recurrent_output, _ = model.rnn(model.encode_inputs(inputs.to(device)))
        predictions = model.predict_from_hidden(recurrent_output)
        logits = predictions['tube_logits_t1']
        if model.predict_coordinates and not relative:
            for name, head in (("current", "coords_t0"), ("next", "coords_t1")):
                truth = np.zeros((*mask.shape, 3), dtype=np.float32)
                for index, record in enumerate(batch_records):
                    truth[index, :len(record.targets)] = getattr(record, f"coords_{name}")
                coordinate_stats[name].update(predictions[head][torch.from_numpy(mask).to(device)].cpu().numpy(), truth[mask])
        elif relative:
            valid_mask = torch.from_numpy(mask).to(device)
            tube_correct = logits[valid_mask].argmax(-1) == torch.from_numpy(targets[mask]).to(device)
            for name, step, label in (('current', 't0', 'current'), ('next', 't1', 'next_relative_to_true_tube')):
                truth = np.zeros((*mask.shape, 3), dtype=np.float32)
                measurements = np.zeros((*mask.shape, 5), dtype=np.float32)
                for index, record in enumerate(batch_records):
                    truth[index, :len(record.targets)] = getattr(record, f'coords_{name}')
                    measurements[index, :len(record.targets)] = record.inputs if name == 'current' else record.measurements_next
                actual = torch.from_numpy(truth[mask]).to(device)
                measured = torch.from_numpy(measurements[mask]).to(device)
                along = predictions[f'along_{step}'][valid_mask]
                lr_logits = predictions[f'lr_logits_{step}'][valid_mask]
                relative_stats[label].update(along, lr_logits, actual, measured)
                for station in measured[..., 4].unique():
                    selected = measured[..., 4] == station
                    relative_station_stats[(label, int(station))].update(along[selected], lr_logits[selected], actual[selected], measured[selected])
                if name == 'next':
                    relative_stats['next_when_tube_correct'].update(along[tube_correct], lr_logits[tube_correct], actual[tube_correct], measured[tube_correct])
        targets_device = torch.from_numpy(targets).to(device)
        mask_device = torch.from_numpy(mask).to(device)

        valid_logits = logits[mask_device]
        valid_targets = targets_device[mask_device]
        nll = F.cross_entropy(valid_logits, valid_targets, reduction="none")
        true_logits = valid_logits.gather(1, valid_targets[:, None]).squeeze(1)
        ranks = (valid_logits > true_logits[:, None]).sum(dim=1) + 1
        top10 = valid_logits.topk(max(TOP_K), dim=1).indices
        topk_correct_t = {
            k: (top10[:, :k] == valid_targets[:, None]).any(dim=1)
            for k in TOP_K
        }

        valid_target_stations = torch.from_numpy(target_stations[mask]).to(device)
        predicted_stations = class_to_station[valid_logits.argmax(dim=1)]
        station_correct += int((predicted_stations == valid_target_stations).sum().item())

        station_nll, station_ranks, station_topk_correct_t = station_masked_metrics(
            valid_logits,
            valid_targets,
            valid_target_stations,
            schema,
        )

        nll_np = nll.cpu().numpy().astype(np.float64, copy=False)
        ranks_np = ranks.cpu().numpy().astype(np.float64, copy=False)
        topk_correct = {k: value.cpu().numpy() for k, value in topk_correct_t.items()}
        station_nll_np = station_nll.cpu().numpy().astype(np.float64, copy=False)
        station_ranks_np = station_ranks.cpu().numpy().astype(np.float64, copy=False)
        station_topk_correct = {
            k: value.cpu().numpy() for k, value in station_topk_correct_t.items()
        }
        flat_targets = targets[mask]
        flat_source = source_stations[mask]
        flat_target_station = target_stations[mask]
        absolute_steps = np.broadcast_to(np.arange(1, max_len + 1), mask.shape)[mask]
        flat_steps = absolute_steps - warmup_steps
        flat_target_hit_numbers = absolute_steps + 1
        flat_track_hits = np.repeat(target_lengths + 1, supervised_lengths)

        overall.update(nll_np, ranks_np, topk_correct)
        station_masked_overall.update(
            station_nll_np, station_ranks_np, station_topk_correct
        )
        for dimension, values in (
            ("prediction_step", flat_steps),
            ("target_hit_number", flat_target_hit_numbers),
            ("source_station", flat_source),
            ("target_station", flat_target_station),
            ("track_hits", flat_track_hits),
        ):
            for value in np.unique(values):
                group_mask = values == value
                grouped_stats[(dimension, str(int(value)))].update(
                    nll_np, ranks_np, topk_correct, group_mask
                )
                grouped_stats[(f"station_masked_{dimension}", str(int(value)))].update(
                    station_nll_np,
                    station_ranks_np,
                    station_topk_correct,
                    group_mask,
                )

        transitions = flat_source * 100 + flat_target_station
        for transition in np.unique(transitions):
            group_mask = transitions == transition
            source = int(transition // 100)
            target = int(transition % 100)
            grouped_stats[("station_transition", f"{source}->{target}")].update(
                nll_np, ranks_np, topk_correct, group_mask
            )

        for tube_id in np.unique(flat_targets):
            tube_stats[int(tube_id)].update(
                nll_np, ranks_np, topk_correct, flat_targets == tube_id
            )

        offset = 0
        for length in supervised_lengths:
            end = offset + int(length)
            track_top1_recall_sum += float(topk_correct[1][offset:end].mean())
            for k in TOP_K:
                track_exact[k] += int(topk_correct[k][offset:end].all())
            offset = end
        tracks += len(batch_records)
        progress.set_postfix(tracks=tracks, top1=f"{overall.as_dict()['top1_recall']:.4f}")

    if tracks == 0 or overall.support == 0:
        raise ValueError("No evaluable tracks were found in the dataset.")

    grouped_rows = group_rows(grouped_stats)
    tube_rows = []
    for class_id, stats in enumerate(tube_stats):
        if stats.support == 0:
            continue
        tube_rows.append(
            {
                "class_id": class_id,
                "station": int(class_to_station_np[class_id]),
                "local_tube_id": int(class_to_local_np[class_id]),
                **stats.as_dict(),
            }
        )

    supported_tubes = [stats for stats in tube_stats if stats.support]
    macro_topk = {
        f"macro_top{k}_recall_by_tube": float(
            np.mean([stats.topk_hits[k] / stats.support for stats in supported_tubes])
        )
        for k in TOP_K
    }
    summary = {
        "dataset": {
            "path": str(args.data.resolve()),
            "size_bytes": args.data.stat().st_size,
            "sha256": dataset_sha256,
            "schema_version": schema.version,
            "num_tubes": schema.num_tubes,
            "station_offsets": schema.station_offsets.tolist(),
            "station_tube_counts": schema.station_tube_counts.tolist(),
            **(
                {
                    "metadata_path": str(schema.metadata_path.resolve()),
                    "metadata_sha256": sha256(schema.metadata_path),
                }
                if schema.metadata_path is not None
                else {}
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
            **checkpoint_info,
        },
        "evaluation": {
            "device": str(device),
            "batch_size": args.batch_size,
            "min_hits": args.min_hits,
            "max_hits": args.max_hits,
            "seed_hits": seed_hits,
            "tracks": tracks,
            "target_station_top1_accuracy": station_correct / overall.support,
            "mean_track_top1_recall": track_top1_recall_sum / tracks,
            **{f"track_exact_top{k}": track_exact[k] / tracks for k in TOP_K},
            **overall.as_dict(),
            **{
                f"station_masked_{key}": value
                for key, value in station_masked_overall.as_dict().items()
            },
            **macro_topk,
            **({"coordinates": {name: stats.as_dict() for name, stats in coordinate_stats.items()}}
               if model.predict_coordinates and not relative else {}),
            **({"tube_relative": {
                "interpretation": "Along/LR targets are relative to the true hit tube. Next XYZ uses its observed dr only in evaluation, never as a network input; it is conditional reconstruction, not unconditional future XYZ prediction.",
                **{name: stats.as_dict() for name, stats in relative_stats.items()},
                "by_station": [{"head": name, "station": station, **stats.as_dict()}
                               for (name, station), stats in sorted(relative_station_stats.items())],
            }} if relative else {}),
        },
        "by_group": grouped_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    write_csv(args.output_dir / "metrics_by_group.csv", grouped_rows)
    write_csv(args.output_dir / "metrics_by_tube.csv", tube_rows)
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary["evaluation"], indent=2, ensure_ascii=False))
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
