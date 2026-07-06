#!/usr/bin/env python
"""Preprocess drift-sim TSV files into fast tensor shards for training."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


DRIFT_SIM_COLUMNS = [
    "ev_id", "wireid", "dr", "lr", "station", "tr_id",
    "x", "y", "x0", "y0", "z0",
]


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


def iter_complete_events(file: Path, chunk_size: int):
    carry = None
    for chunk in pd.read_csv(
        file,
        sep=r"\s+",
        names=DRIFT_SIM_COLUMNS,
        engine="c",
        chunksize=chunk_size,
    ):
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
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--shard-size", type=int, default=200_000)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-hits", type=int, default=8)
    parser.add_argument("--num-stations", type=int, default=8)
    parser.add_argument("--tubes-per-station", type=int, default=151)
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
        progress = tqdm(iter_complete_events(file, args.chunk_size), desc=f"Preprocessing {file.name}")
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
        "input_dir": str(input_dir),
        "file_pattern": args.file_pattern,
        "validation_split": args.validation_split,
        "split_seed": args.split_seed,
        "input_columns": args.input_columns,
        "num_stations": args.num_stations,
        "tubes_per_station": args.tubes_per_station,
        "train_tracks": train_writer.total_tracks,
        "validation_tracks": val_writer.total_tracks,
    }
    torch.save(metadata, output_dir / "metadata.pt")
    print(f"Saved cache to: {output_dir}")
    print(f"Train tracks: {train_writer.total_tracks}")
    print(f"Validation tracks: {val_writer.total_tracks}")


if __name__ == "__main__":
    main()
