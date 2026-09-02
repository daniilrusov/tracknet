import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / "mplconfig"),
)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import driftsim_v3
import evaluate_straw_checkpoint
from src.tracknet.data.dataset import PrebatchedStrawTracksDataset
from src.tracknet.training import StrawTrackNETModule


class DatasetSchemaTests(unittest.TestCase):
    def test_v3_schema_uses_variable_station_ranges(self):
        config = driftsim_v3.default_config()
        config["events"] = 3
        geometry = driftsim_v3.build_detector_geometry(config["detector"])
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            data_path = output_dir / "output.tsv"
            rows = driftsim_v3.write_data(data_path, config, geometry)
            driftsim_v3.save_yaml(
                output_dir / "metadata.yaml",
                driftsim_v3.metadata(config, geometry, data_path, rows),
            )
            args = argparse.Namespace(
                data=data_path,
                schema_version="auto",
                metadata=None,
                num_stations=8,
                tubes_per_station=151,
            )
            schema = evaluate_straw_checkpoint.load_dataset_schema(args)
            self.assertEqual(schema.version, "v3")
            self.assertEqual(schema.num_tubes, 1456)
            np.testing.assert_array_equal(
                schema.station_tube_counts,
                [151, 151, 213, 213, 151, 151, 213, 213],
            )
            np.testing.assert_array_equal(
                schema.station_offsets,
                [0, 151, 302, 515, 728, 879, 1030, 1243],
            )

    def test_station_masked_metrics_respect_unequal_ranges(self):
        schema = evaluate_straw_checkpoint.DatasetSchema(
            version="v3",
            num_tubes=5,
            station_offsets=np.array([0, 2], dtype=np.int64),
            station_tube_counts=np.array([2, 3], dtype=np.int64),
        )
        logits = torch.tensor(
            [[0.0, 4.0, 20.0, 0.0, 0.0], [20.0, 0.0, 0.0, 1.0, 5.0]]
        )
        targets = torch.tensor([1, 4])
        stations = torch.tensor([1, 2])
        _, ranks, topk = evaluate_straw_checkpoint.station_masked_metrics(
            logits, targets, stations, schema
        )
        torch.testing.assert_close(ranks, torch.ones(2, dtype=torch.int64))
        self.assertTrue(bool(topk[1].all()))


class _SingleBatchDataset(IterableDataset):
    def __iter__(self):
        yield {
            "inputs": torch.zeros(2, 2, 5),
            "targets": torch.ones(2, 2, dtype=torch.long),
            "target_mask": torch.ones(2, 2, dtype=torch.bool),
            "input_lengths": [2, 2],
        }


class _EpochControlledModule(StrawTrackNETModule):
    def __init__(self):
        super().__init__(
            input_features=5,
            hidden_features=4,
            num_tubes=4,
        )

    def forward(self, batch):
        prediction = 1 if self.current_epoch == 0 else 2
        shape = (*batch["targets"].shape, 4)
        predicted_ids = torch.full(
            batch["targets"].shape,
            prediction,
            dtype=torch.long,
            device=batch["targets"].device,
        )
        logits = F.one_hot(predicted_ids, num_classes=shape[-1]).float() * 10.0
        logits = logits + next(self.model.parameters()).sum() * 0.0
        return {"tube_logits_t1": logits}


class _ValidationHistory(pl.Callback):
    def __init__(self):
        self.values = []

    def on_validation_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("val_hit_efficiency_t1")
        if value is not None:
            self.values.append(float(value))


class ValidationMetricLifecycleTests(unittest.TestCase):
    def test_validation_metric_is_reset_between_epochs(self):
        history = _ValidationHistory()
        loader = torch.utils.data.DataLoader(_SingleBatchDataset(), batch_size=None)
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            callbacks=[history],
        )
        trainer.fit(_EpochControlledModule(), loader, loader)
        self.assertEqual(len(history.values), 2)
        self.assertAlmostEqual(history.values[0], 1.0, places=6)
        self.assertAlmostEqual(history.values[1], 0.0, places=6)


class PrebatchedShuffleTests(unittest.TestCase):
    def test_training_order_changes_each_iteration_but_validation_does_not(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            for split in ("train", "validation"):
                split_dir = cache_dir / split
                split_dir.mkdir()
                inputs = torch.zeros(12, 2, 5)
                inputs[:, 0, 0] = torch.arange(12)
                torch.save(
                    {
                        "inputs": inputs,
                        "targets": torch.zeros(12, 2, dtype=torch.int16),
                        "target_mask": torch.ones(12, 2, dtype=torch.bool),
                        "input_lengths": torch.full((12,), 2, dtype=torch.int16),
                    },
                    split_dir / "shard_000000.pt",
                )

            def order(dataset):
                return [
                    int(value)
                    for batch in dataset
                    for value in batch["inputs"][:, 0, 0].tolist()
                ]

            train = PrebatchedStrawTracksDataset(
                cache_dir, batch_size=4, split="train", shuffle_tracks=True
            )
            first_train = order(train)
            second_train = order(train)
            self.assertEqual(sorted(first_train), list(range(12)))
            self.assertEqual(sorted(second_train), list(range(12)))
            self.assertNotEqual(first_train, second_train)

            validation = PrebatchedStrawTracksDataset(
                cache_dir, batch_size=4, split="validation", shuffle_tracks=True
            )
            self.assertEqual(order(validation), order(validation))


if __name__ == "__main__":
    unittest.main()
