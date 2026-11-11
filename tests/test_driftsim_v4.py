import csv
import math
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import driftsim_v3
import driftsim_v4


class DriftSimV4Tests(unittest.TestCase):
    def test_full_misses_are_retried_before_applying_hit_efficiency(self):
        config = driftsim_v4.default_config()
        config["events"] = 1
        geometry = driftsim_v3.build_detector_geometry(config["detector"])
        miss = (100.0, 0.0, math.cos(math.pi / 4))
        partial = (200.0, 0.0, math.cos(math.atan(0.6)))
        through = (300.0, 0.0, 1.0)
        draws = (0.0, 0.0, 2.0, *miss, *partial, *miss, *through)
        expected = [(0, 1), (0, 2)] + [(1, station) for station in range(1, 9)]

        for efficiency in (1.0, 0.0):
            with self.subTest(efficiency=efficiency):
                config["simulation"]["efficiency"] = efficiency
                rng = Mock(spec=random.Random)
                rng.uniform.side_effect = draws
                rng.random.return_value = 0.5
                rows = list(driftsim_v3.iter_rows(config, geometry, rng, resample_misses=True))

                self.assertEqual(rng.uniform.call_count, len(draws))
                self.assertEqual(
                    [(row["tr_id"], row["station"]) for row in rows],
                    expected if efficiency == 1.0 else [],
                )

    def test_v4_entry_point_writes_requested_multiplicity_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config_path = output / "input.yaml"
            config_path.write_text(
                "simulation:\n  tracks_per_event_min: 5\n  tracks_per_event_max_exclusive: 6\n",
                encoding="utf-8",
            )
            argv = ["driftsim_v4.py", "-c", str(config_path), "-o", str(output), "-n", "20"]
            with patch.object(sys, "argv", argv):
                driftsim_v4.main()

            metadata = yaml.safe_load((output / "metadata.yaml").read_text(encoding="utf-8"))
            self.assertEqual(metadata["generator_version"], 4)
            self.assertEqual(metadata["schema_version"], 3)
            stations = metadata["detector"]["stations"]
            np.testing.assert_array_equal(
                [station["z_mm"] for station in stations], np.linspace(1200.0, 1440.0, 8)
            )
            tracks = defaultdict(set)
            with (output / "output.tsv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    tracks[int(row["ev_id"])].add(int(row["tr_id"]))
                    self.assertEqual(float(row["z"]), stations[int(row["station"]) - 1]["z_mm"])
                    self.assertEqual(row["z"], row["z0"])
            self.assertEqual(tracks, {event_id: set(range(5)) for event_id in range(20)})


if __name__ == "__main__":
    unittest.main()
