import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import driftsim_v2
import driftsim_v3
import preprocess_drift_sim


class DriftSimV3GeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = driftsim_v3.default_config()
        self.geometry = driftsim_v3.build_detector_geometry(self.config["detector"])

    def test_plane_counts_and_dense_offsets(self):
        counts = [plane.tube_count for plane in self.geometry.planes]
        offsets = [plane.class_offset for plane in self.geometry.planes]
        self.assertEqual(counts, [151, 151, 213, 213, 151, 151, 213, 213])
        self.assertEqual(offsets, [0, 151, 302, 515, 728, 879, 1030, 1243])
        self.assertEqual(self.geometry.total_tubes, 1456)

        class_ids = []
        for plane in self.geometry.planes:
            class_ids.extend(range(plane.class_offset, plane.class_offset + plane.tube_count))
        self.assertEqual(class_ids, list(range(self.geometry.total_tubes)))

    def test_measurements_are_bounded_and_round_trip_to_wire_centers(self):
        samples = np.linspace(-749.999, 749.999, 17)
        for plane in self.geometry.planes:
            for x in samples:
                for y in samples:
                    measurement = plane.measurement(float(x), float(y))
                    local_id = int(measurement["local_tube_id"])
                    self.assertGreaterEqual(local_id, 0)
                    self.assertLess(local_id, plane.tube_count)
                    self.assertLessEqual(float(measurement["dr"]), plane.pitch / 2 + 1e-9)
                    self.assertEqual(
                        int(measurement["wireid"]), plane.station * 1000 + local_id
                    )
                    anchor = np.array([measurement["x0"], measurement["y0"]])
                    self.assertAlmostEqual(
                        float(plane.normal @ anchor),
                        float(measurement["wire_coordinate"]),
                        places=9,
                    )
                    self.assertLessEqual(float(np.abs(anchor).max()), 750.0 + 1e-9)

    def test_z_planes_are_shared_with_truth(self):
        expected = np.linspace(0.0, 240.0, 8)
        actual = np.array([plane.z for plane in self.geometry.planes])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        for plane in self.geometry.planes:
            measurement = plane.measurement(0.0, 0.0)
            self.assertEqual(float(measurement["z0"]), plane.z)

    def test_truth_extrapolation_preserves_legacy_physics(self):
        parameters = [
            (100.0, -200.0, 0.2, 1.3, 0.0),
            (-400.0, 300.0, 0.7, 5.1, 102.85714285714286),
            (10.0, 20.0, 1.0, 2.2, 240.0),
        ]
        for vertex_x, vertex_y, theta, phi, z in parameters:
            legacy = driftsim_v2.LineExtrapToZ(vertex_x, vertex_y, theta, phi, z)
            v3 = driftsim_v3.line_extrapolate_to_z(
                vertex_x, vertex_y, theta, phi, z
            )
            np.testing.assert_allclose(v3, legacy, rtol=0.0, atol=1e-12)


class DriftSimV3OutputTests(unittest.TestCase):
    def test_output_is_deterministic_and_preprocessor_validates_ids(self):
        config = driftsim_v3.default_config()
        config["events"] = 20
        config["seed"] = 1234
        geometry = driftsim_v3.build_detector_geometry(config["detector"])

        with tempfile.TemporaryDirectory() as temporary_dir:
            first = Path(temporary_dir) / "first.tsv"
            second = Path(temporary_dir) / "second.tsv"
            first_rows = driftsim_v3.write_data(first, config, geometry)
            second_rows = driftsim_v3.write_data(second, config, geometry)
            self.assertGreater(first_rows, 0)
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            frame = pd.read_csv(first, sep="\t")
            self.assertEqual(list(frame.columns), driftsim_v3.OUTPUT_COLUMNS)
            self.assertTrue((frame["schema_version"] == 3).all())
            self.assertTrue((frame["x"].abs() < 750.0).all())
            self.assertTrue((frame["y"].abs() < 750.0).all())
            self.assertTrue((frame["x0"].abs() <= 750.0 + 1e-9).all())
            self.assertTrue((frame["y0"].abs() <= 750.0 + 1e-9).all())
            self.assertTrue(np.allclose(frame["z"], frame["z0"]))
            self.assertTrue((frame["dr"] <= 5.0 + 1e-9).all())
            self.assertTrue((frame["tube_class_id"] >= 0).all())
            self.assertTrue((frame["tube_class_id"] < geometry.total_tubes).all())
            expected_wire_ids = frame["station"] * 1000 + frame["local_tube_id"]
            self.assertTrue((frame["wireid"] == expected_wire_ids).all())

            event_iterator = preprocess_drift_sim.iter_complete_events(
                first, chunk_size=200, schema_version="v3"
            )
            try:
                _, event = next(event_iterator)
                offsets = np.array(
                    [plane.class_offset for plane in geometry.planes], dtype=np.int64
                )
                counts = np.array(
                    [plane.tube_count for plane in geometry.planes], dtype=np.int64
                )
                class_ids = preprocess_drift_sim.v3_tube_class_ids(
                    event, geometry.total_tubes, offsets, counts
                )
                np.testing.assert_array_equal(
                    class_ids.astype(np.int64),
                    event["tube_class_id"].to_numpy(dtype=np.int64),
                )

                broken = event.copy()
                row_index = broken.index[0]
                broken.loc[row_index, "tube_class_id"] = (
                    int(broken.loc[row_index, "tube_class_id"]) + 1
                ) % geometry.total_tubes
                with self.assertRaisesRegex(ValueError, "Inconsistent V3 tube class"):
                    preprocess_drift_sim.v3_tube_class_ids(
                        broken, geometry.total_tubes, offsets, counts
                    )
            finally:
                event_iterator.close()


if __name__ == "__main__":
    unittest.main()
