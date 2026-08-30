#!/usr/bin/env python
"""Preprocess drift-sim TSV files into fast tensor shards for training."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


DRIFT_SIM_COLUMNS = [
    "ev_id", "wireid", "dr", "lr", "station", "tr_id",
    "x", "y", "x0", "y0", "z0",
]
V3_REQUIRED_COLUMNS = {
    "schema_version", "ev_id", "tr_id", "station", "wireid",
    "local_tube_id", "tube_class_id", "x0", "y0", "z0", "dr", "lr",
}


def event_hash_int(event_id, seed: int) -> int:
    payload = f"split:{seed}:{event_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def is_validation_event(event_id, validation_split: float, seed: int) -> bool:
    return event_hash_int(event_id, seed) / 2**64 < validation_split


def tube_class_ids(
    group: pd.DataFrame,
    num_stations: int,
    tubes_per_station: int,
    tube_id_offset: int,
    tube_id_mapping: str,
) -> np.ndarray:
    station = group["station"].astype(np.int64).to_numpy()
    if np.any(station < 1) or np.any(station > num_stations):
        bad_station = int(station[(station < 1) | (station > num_stations)][0])
        raise ValueError(f"Station id {bad_station} is outside [1, {num_stations}].")

    raw_tube_id = group["wireid"].astype(np.int64).to_numpy() % 1000
    tube_id = raw_tube_id - tube_id_offset
    if tube_id_mapping == "station_modulo":
        tube_id = np.mod(tube_id, tubes_per_station)
    elif np.any((tube_id < 0) | (tube_id >= tubes_per_station)):
        bad_tube = int(tube_id[(tube_id < 0) | (tube_id >= tubes_per_station)][0])
        raise ValueError(
            f"Tube id {bad_tube} is outside [0, {tubes_per_station}). "
            "Use tube_id_mapping='station_modulo' or adjust tube_id_offset."
        )

    return ((station - 1) * tubes_per_station + tube_id).astype(np.int16)


def v3_tube_class_ids(
    group: pd.DataFrame,
    num_tubes: int,
    station_offsets: np.ndarray | None = None,
    station_tube_counts: np.ndarray | None = None,
) -> np.ndarray:
    missing = V3_REQUIRED_COLUMNS.difference(group.columns)
    if missing:
        raise ValueError(f"V3 data is missing columns: {sorted(missing)}")
    schema_versions = group["schema_version"].astype(np.int64).to_numpy()
    if np.any(schema_versions != 3):
        bad_version = int(schema_versions[schema_versions != 3][0])
        raise ValueError(f"Expected schema_version=3, got {bad_version}.")

    stations = group["station"].astype(np.int64).to_numpy()
    local_ids = group["local_tube_id"].astype(np.int64).to_numpy()
    wire_ids = group["wireid"].astype(np.int64).to_numpy()
    class_ids = group["tube_class_id"].astype(np.int64).to_numpy()
    expected_wire_ids = stations * 1000 + local_ids
    if np.any(stations < 1):
        bad_station = int(stations[stations < 1][0])
        raise ValueError(f"V3 station {bad_station} must be positive.")
    if np.any(local_ids < 0):
        bad_local_id = int(local_ids[local_ids < 0][0])
        raise ValueError(f"V3 local_tube_id {bad_local_id} must be non-negative.")
    if np.any(wire_ids != expected_wire_ids):
        index = int(np.flatnonzero(wire_ids != expected_wire_ids)[0])
        raise ValueError(
            "Inconsistent V3 wire id: expected "
            f"{expected_wire_ids[index]}, got {wire_ids[index]}."
        )
    if np.any(class_ids < 0) or np.any(class_ids >= num_tubes):
        bad_class = int(class_ids[(class_ids < 0) | (class_ids >= num_tubes)][0])
        raise ValueError(f"V3 tube_class_id {bad_class} is outside [0, {num_tubes}).")

    if station_offsets is not None and station_tube_counts is not None:
        station_indexes = stations - 1
        if np.any(station_indexes >= len(station_offsets)):
            bad_station = int(stations[station_indexes >= len(station_offsets)][0])
            raise ValueError(
                f"V3 station {bad_station} is outside [1, {len(station_offsets)}]."
            )
        allowed_counts = station_tube_counts[station_indexes]
        if np.any(local_ids >= allowed_counts):
            index = int(np.flatnonzero(local_ids >= allowed_counts)[0])
            raise ValueError(
                f"V3 local_tube_id {local_ids[index]} is outside station "
                f"{stations[index]} range [0, {allowed_counts[index]})."
            )
        expected_class_ids = station_offsets[station_indexes] + local_ids
        if np.any(class_ids != expected_class_ids):
            index = int(np.flatnonzero(class_ids != expected_class_ids)[0])
            raise ValueError(
                f"Inconsistent V3 tube class: expected {expected_class_ids[index]}, "
                f"got {class_ids[index]}."
            )
    return class_ids.astype(np.int16)


class SplitShardWriter:
    def __init__(self, output_dir: Path, split: str, shard_size: int, max_len: int, n_features: int):
        self.output_dir = output_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.max_len = max_len
        self.n_features = n_features
        self.shard_index = 0
        self.inputs = []
        self.targets = []
        self.target_masks = []
        self.input_lengths = []
        self.total_tracks = 0

    def add(self, hits: np.ndarray, tube_ids: np.ndarray):
        input_len = len(hits) - 1
        inputs = np.zeros((self.max_len, self.n_features), dtype=np.float32)
        targets = np.zeros((self.max_len,), dtype=np.int16)
        target_mask = np.zeros((self.max_len,), dtype=np.bool_)

        inputs[:input_len] = hits[:-1].astype(np.float32, copy=False)
        targets[:input_len] = tube_ids[1:].astype(np.int16, copy=False)
        target_mask[:input_len] = True

        self.inputs.append(inputs)
        self.targets.append(targets)
        self.target_masks.append(target_mask)
        self.input_lengths.append(input_len)
        self.total_tracks += 1

        if len(self.inputs) >= self.shard_size:
            self.flush()

    def flush(self):
        if len(self.inputs) == 0:
            return

        shard_path = self.output_dir / f"shard_{self.shard_index:06d}.pt"
        torch.save(
            {
                "inputs": torch.from_numpy(np.stack(self.inputs)),
                "targets": torch.from_numpy(np.stack(self.targets)),
                "target_mask": torch.from_numpy(np.stack(self.target_masks)),
                "input_lengths": torch.tensor(self.input_lengths, dtype=torch.int16),
            },
            shard_path,
        )

        self.shard_index += 1
        self.inputs.clear()
        self.targets.clear()
        self.target_masks.clear()
        self.input_lengths.clear()


def iter_complete_events(file: Path, chunk_size: int, schema_version: str = "legacy"):
    read_kwargs = {
        "filepath_or_buffer": file,
        "sep": r"\s+",
        "engine": "c",
        "chunksize": chunk_size,
    }
    if schema_version == "legacy":
        read_kwargs["names"] = DRIFT_SIM_COLUMNS
    else:
        read_kwargs["header"] = 0
    carry = None
    reader = pd.read_csv(**read_kwargs)
    try:
        for chunk in reader:
            if schema_version == "v3":
                missing = V3_REQUIRED_COLUMNS.difference(chunk.columns)
                if missing:
                    raise ValueError(f"V3 data is missing columns: {sorted(missing)}")
            if carry is not None and len(carry) > 0:
                chunk = pd.concat([carry, chunk], ignore_index=True)

            last_event_id = chunk["ev_id"].iloc[-1]
            complete = chunk[chunk["ev_id"] != last_event_id]
            carry = chunk[chunk["ev_id"] == last_event_id]

            if len(complete) == 0:
                continue

            for event_id, event in complete.groupby("ev_id", sort=False):
                yield event_id, event

        if carry is not None and len(carry) > 0:
            for event_id, event in carry.groupby("ev_id", sort=False):
                yield event_id, event
    finally:
        reader.close()


def process_event(event, writer, args):
    for _, group in event.groupby("tr_id", sort=False):
        if "station" in group.columns:
            group = group.sort_values("station")
        elif "z0" in group.columns:
            group = group.sort_values("z0")

        n_hits = len(group)
        if n_hits < args.min_hits or n_hits > args.max_hits:
            continue

        hits = group[args.input_columns].to_numpy(dtype=np.float32, copy=True)
        if args.schema_version == "v3":
            tube_ids = v3_tube_class_ids(
                group,
                args.num_tubes,
                args.v3_station_offsets,
                args.v3_station_tube_counts,
            )
        else:
            tube_ids = tube_class_ids(
                group,
                num_stations=args.num_stations,
                tubes_per_station=args.tubes_per_station,
                tube_id_offset=args.tube_id_offset,
                tube_id_mapping=args.tube_id_mapping,
            )
        writer.add(hits, tube_ids)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess drift-sim TSV files into fast tensor shards."
    )
    parser.add_argument("--input-dir", default="outputs/drift_sim")
    parser.add_argument("--output-dir", default="outputs/drift_sim_cache")
    parser.add_argument("--file-pattern", default="*.tsv")
    parser.add_argument(
        "--schema-version",
        choices=["legacy", "v3"],
        default="legacy",
        help="Legacy headerless V1/V2 data or headered V3 data.",
    )
    parser.add_argument("--metadata-filename", default="metadata.yaml")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--shard-size", type=int, default=200_000)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-hits", type=int, default=8)
    parser.add_argument("--num-stations", type=int, default=8)
    parser.add_argument("--tubes-per-station", type=int, default=151)
    parser.add_argument("--num-tubes", type=int)
    parser.add_argument("--tube-id-offset", type=int, default=0)
    parser.add_argument("--tube-id-mapping", default="station_modulo")
    parser.add_argument(
        "--input-columns",
        nargs="+",
        default=["x0", "y0", "z0", "dr", "lr", "station"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob(args.file_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matching '{args.file_pattern}' in {input_dir}")

    source_metadata = None
    args.v3_station_offsets = None
    args.v3_station_tube_counts = None
    if args.schema_version == "v3":
        metadata_path = input_dir / args.metadata_filename
        if not metadata_path.exists():
            raise FileNotFoundError(f"V3 metadata not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            source_metadata = yaml.safe_load(stream) or {}
        if int(source_metadata.get("schema_version", -1)) != 3:
            raise ValueError(f"Expected schema_version=3 in {metadata_path}.")
        stations_metadata = source_metadata.get("detector", {}).get("stations", [])
        if not stations_metadata:
            raise ValueError(f"V3 station geometry is missing from {metadata_path}.")
        expected_station_numbers = list(range(1, len(stations_metadata) + 1))
        station_numbers = [int(station["station"]) for station in stations_metadata]
        if station_numbers != expected_station_numbers:
            raise ValueError(
                "V3 station metadata must be ordered and numbered consecutively "
                f"from 1; got {station_numbers}."
            )
        args.v3_station_offsets = np.asarray(
            [int(station["class_offset"]) for station in stations_metadata],
            dtype=np.int64,
        )
        args.v3_station_tube_counts = np.asarray(
            [int(station["tube_count"]) for station in stations_metadata],
            dtype=np.int64,
        )
        expected_offsets = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.cumsum(args.v3_station_tube_counts[:-1]),
            )
        )
        if np.any(args.v3_station_tube_counts <= 0):
            raise ValueError("V3 station tube counts must be positive.")
        if not np.array_equal(args.v3_station_offsets, expected_offsets):
            raise ValueError(
                "V3 class offsets must form one contiguous dense class range."
            )
        metadata_num_tubes = int(source_metadata["total_tubes"])
        if int(args.v3_station_tube_counts.sum()) != metadata_num_tubes:
            raise ValueError(
                "V3 station tube counts do not sum to metadata total_tubes."
            )
        args.num_stations = len(stations_metadata)
        if args.num_tubes is None:
            args.num_tubes = metadata_num_tubes
        elif args.num_tubes != metadata_num_tubes:
            raise ValueError(
                f"--num-tubes={args.num_tubes} does not match V3 metadata "
                f"total_tubes={metadata_num_tubes}."
            )
    elif args.num_tubes is None:
        args.num_tubes = args.num_stations * args.tubes_per_station

    for split in ("train", "validation"):
        split_dir = output_dir / split
        if split_dir.exists():
            for shard_file in split_dir.glob("*.pt"):
                shard_file.unlink()
    metadata_path = output_dir / "metadata.pt"
    if metadata_path.exists():
        metadata_path.unlink()

    max_len = args.max_hits - 1
    n_features = len(args.input_columns)
    train_writer = SplitShardWriter(output_dir, "train", args.shard_size, max_len, n_features)
    val_writer = SplitShardWriter(output_dir, "validation", args.shard_size, max_len, n_features)

    for file in files:
        progress = tqdm(
            iter_complete_events(file, args.chunk_size, args.schema_version),
            desc=f"Preprocessing {file.name}",
        )
        for event_id, event in progress:
            writer = (
                val_writer
                if is_validation_event(event_id, args.validation_split, args.split_seed)
                else train_writer
            )
            process_event(event, writer, args)
            progress.set_postfix(
                train=train_writer.total_tracks,
                validation=val_writer.total_tracks,
            )

    train_writer.flush()
    val_writer.flush()

    metadata = {
        "schema_version": args.schema_version,
        "input_dir": str(input_dir),
        "file_pattern": args.file_pattern,
        "validation_split": args.validation_split,
        "split_seed": args.split_seed,
        "input_columns": args.input_columns,
        "num_stations": args.num_stations,
        "tubes_per_station": args.tubes_per_station,
        "num_tubes": args.num_tubes,
        "train_tracks": train_writer.total_tracks,
        "validation_tracks": val_writer.total_tracks,
    }
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata
    torch.save(metadata, output_dir / "metadata.pt")
    print(f"Saved cache to: {output_dir}")
    print(f"Train tracks: {train_writer.total_tracks}")
    print(f"Validation tracks: {val_writer.total_tracks}")


if __name__ == "__main__":
    main()
