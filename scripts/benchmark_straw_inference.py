#!/usr/bin/env python
"""Run event-level StrawTrackNET inference on a DriftSim V3 test TSV.

An exact match requires the same hit rows as a truth track. Precision and
efficiency divide exact matches by the number of candidates and truth tracks,
respectively. Only tracks with at least --min-hits enter the metrics.
Latency covers one inference call, including seed construction and branching.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from itertools import chain, islice
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_straw_checkpoint import GEOMETRY_INPUT_COLUMNS, iter_complete_events, load_model
from src.tracknet.inference import StrawTrackNETInference
from src.tracknet.straw_geometry import V3_TUBE_COUNTS


def score_event(
    event: pd.DataFrame, candidates: list[list[int]], min_hits: int,
) -> tuple[int, int, int]:
    truth = {
        tuple(rows)
        for track_id, rows in event.groupby("tr_id", sort=False).indices.items()
        if track_id >= 0 and len(rows) >= min_hits
    }
    selected = [tuple(sorted(path)) for path in candidates if len(path) >= min_hits]
    return len(truth), len(selected), len(truth.intersection(selected))


def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use a CPU or CUDA device for this benchmark.")

    required = {*GEOMETRY_INPUT_COLUMNS, "ev_id", "tr_id", "tube_class_id"}
    columns = pd.read_csv(args.data, sep="\t", nrows=0).columns
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Test TSV is missing columns: {', '.join(sorted(missing))}.")

    model, _ = load_model(args.checkpoint, device)
    if model.input_features != len(GEOMETRY_INPUT_COLUMNS):
        raise ValueError("Event inference requires a checkpoint with V3 hit inputs.")
    infer = StrawTrackNETInference(model, batch_size=args.batch_size)

    latencies = []
    hits = all_candidates = truth_tracks = scored_candidates = exact_matches = 0
    with closing(iter_complete_events(args.data, args.chunk_size, "v3")) as events:
        selected = islice(events, args.max_events)
        warmup = list(islice(selected, args.warmup_events))
        for _, event in warmup:
            infer(event)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for event_id, event in tqdm(chain(warmup, selected), desc="Evaluating events", unit="event"):
            if not event["tube_class_id"].between(0, model.num_tubes - 1).all():
                raise ValueError(f"Event {event_id} has a tube_class_id outside the model's class range.")

            start = perf_counter()
            candidates = infer(event)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(perf_counter() - start)

            n_truth, n_scored, n_matched = score_event(event, candidates, args.min_hits)
            hits += len(event)
            all_candidates += len(candidates)
            truth_tracks += n_truth
            scored_candidates += n_scored
            exact_matches += n_matched

    if not latencies:
        raise ValueError("Test TSV contains no events.")

    elapsed = sum(latencies)
    p50, p95, p99 = np.percentile(np.asarray(latencies) * 1000, [50, 95, 99]).tolist()
    efficiency = exact_matches / truth_tracks if truth_tracks else None
    precision = exact_matches / scored_candidates if scored_candidates else None
    support = truth_tracks + scored_candidates
    return {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "batch_size": args.batch_size,
        "min_hits": args.min_hits,
        "events": len(latencies),
        "hits": hits,
        "metrics": {
            "truth_tracks": truth_tracks,
            "all_candidates": all_candidates,
            "scored_candidates": scored_candidates,
            "exact_matches": exact_matches,
            "track_efficiency": efficiency,
            "track_precision": precision,
            "fake_rate": 1 - precision if precision is not None else None,
            "track_f1": 2 * exact_matches / support if support else None,
        },
        "timing": {
            "warmup_events": len(warmup),
            "inference_seconds": elapsed,
            "events_per_second": len(latencies) / elapsed,
            "latency_ms": {"mean": elapsed * 1000 / len(latencies), "p50": p50, "p95": p95, "p99": p99},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="V3 output.tsv grouped by ev_id.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted V3 Lightning checkpoint.")
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512, help="Candidates per model batch.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="TSV rows per read chunk.")
    parser.add_argument("--min-hits", type=int, default=3, help="Minimum track length for scoring.")
    parser.add_argument("--max-events", type=int, help="Limit the number of measured events.")
    parser.add_argument("--warmup-events", type=int, default=5, help="Run these events once before timing.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        parser.error("--batch-size and --chunk-size must be positive.")
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive.")
    if args.warmup_events < 0:
        parser.error("--warmup-events must be non-negative.")
    if not 2 <= args.min_hits <= len(V3_TUBE_COUNTS):
        parser.error(f"--min-hits must be between 2 and {len(V3_TUBE_COUNTS)}.")

    report = json.dumps(benchmark(args), indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
