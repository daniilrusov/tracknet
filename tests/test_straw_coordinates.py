import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from preprocess_drift_sim import SplitShardWriter
from evaluate_straw_checkpoint import load_model, CoordinateStats
from src.tracknet.data.collate import collate_fn
from src.tracknet.data.dataset import PrebatchedStrawTracksDataset
from src.tracknet.data.schemas import StrawTrack
from src.tracknet.loss import StrawCoordinateLoss
from src.tracknet.metrics import StrawCoordinateRMSE
from src.tracknet.model import StrawTrackNET
from src.tracknet.training import StrawTrackNETModule


def test_coordinates_align_current_and_future_hits_through_shuffled_cache(tmp_path):
    writer = SplitShardWriter(tmp_path, 'train', 4, 4, 5)
    for i, n in enumerate((3, 5, 4, 3)):
        hits = np.zeros((n, 5), dtype=np.float32)
        hits[:, 0] = i
        truth = np.arange(n * 3, dtype=np.float32).reshape(n, 3) + i * 100
        writer.add(hits, np.arange(n), truth)
    dataset = PrebatchedStrawTracksDataset(tmp_path, 2, shuffle_tracks=True, require_coordinate_targets=True)
    for batch in dataset:
        for row, n in enumerate(batch['input_lengths']):
            i = int(batch['inputs'][row, 0, 0])
            expected = torch.arange((n + 1) * 3).reshape(n + 1, 3).float() + i * 100
            torch.testing.assert_close(batch['coords_current'][row, :n], expected[:-1])
            torch.testing.assert_close(batch['coords_next'][row, :n], expected[1:])
            assert batch['target_mask'][row].tolist() == [False] + [True] * (n - 1) + [False] * (4 - n)


def test_coordinate_loss_masks_warmup_padding_and_has_correct_scale():
    prediction = torch.tensor([[[float('nan')] * 3, [3., 4., 12.], [float('nan')] * 3]], requires_grad=True)
    loss = StrawCoordinateLoss((3., 4., 12.))(prediction, torch.zeros_like(prediction), torch.tensor([[False, True, False]]))
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0].count_nonzero() == 0


def test_coordinate_rmse_weights_hits_and_resets():
    metric = StrawCoordinateRMSE()
    metric.update(torch.tensor([[[3., 4., 0.]]]), torch.zeros(1, 1, 3), torch.ones(1, 1, dtype=torch.bool))
    metric.update(torch.zeros(1, 3, 3), torch.zeros(1, 3, 3), torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(metric.compute(), torch.tensor(2.5))
    metric.reset()
    assert metric.count == 0


def test_heads_are_causal_and_receive_gradients_with_tube_classifier():
    torch.manual_seed(42)
    model = StrawTrackNET(hidden_features=8, num_tubes=12, predict_coordinates=True)
    inputs = torch.randn(2, 4, 5)
    inputs[..., 4] = torch.arange(1, 5)
    output = model(inputs, [4, 3])
    changed = inputs.clone()
    changed[:, 2:, :4] += 100
    altered = model(changed, [4, 3])
    for key in output:
        torch.testing.assert_close(output[key][:, :2], altered[key][:, :2])
    mask = torch.tensor([[False, True, True, True], [False, True, True, False]])
    loss_fn = StrawCoordinateLoss()
    loss = loss_fn(output['coords_t0'], torch.zeros(2, 4, 3), mask)
    loss += loss_fn(output['coords_t1'], torch.ones(2, 4, 3), mask)
    loss += torch.nn.functional.cross_entropy(output['tube_logits_t1'][mask], torch.ones(5, dtype=torch.long))
    loss.backward()
    for part in (model.rnn, model.coords_current, model.coords_next, model.tube_classifier):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in part.parameters())
        assert sum(float(p.grad.abs().sum()) for p in part.parameters()) > 0


def test_all_heads_use_same_seed_offset():
    module = StrawTrackNETModule(hidden_features=4, num_tubes=10, predict_coordinates=True)
    output = {'tube_logits_t1': torch.randn(1, 3, 10), 'coords_t0': torch.randn(1, 3, 3), 'coords_t1': torch.randn(1, 3, 3)}
    batch = {'targets': torch.tensor([[1, 2, 3]]), 'target_mask': torch.ones(1, 3, dtype=torch.bool)}
    supervised, targets, _ = module._supervised_view(output, batch)
    for key in output:
        torch.testing.assert_close(supervised[key], output[key][:, 1:])
    assert targets.tolist() == [[2, 3]]


@pytest.mark.parametrize('predict_coordinates', [False, True])
def test_evaluator_loads_both_checkpoint_architectures(tmp_path, predict_coordinates):
    module = StrawTrackNETModule(hidden_features=4, num_tubes=12, predict_coordinates=predict_coordinates)
    path = tmp_path / 'model.ckpt'
    hparams = dict(module.hparams)
    if not predict_coordinates:
        for key in ('predict_coordinates', 'coordinate_center', 'coordinate_scale'):
            hparams.pop(key)
    torch.save({'state_dict': module.state_dict(), 'hyper_parameters': hparams}, path)
    model, _ = load_model(path, torch.device('cpu'))
    assert model.predict_coordinates == predict_coordinates
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, module.model.state_dict()[key])


def test_old_cache_requires_regeneration_when_coordinates_enabled(tmp_path):
    torch.save({'input_columns': ['x0', 'y0', 'z0', 'dr', 'station']}, tmp_path / 'metadata.pt')
    with pytest.raises(ValueError, match='Regenerate'):
        PrebatchedStrawTracksDataset(tmp_path, 2, require_coordinate_targets=True)


def test_raw_collation_uses_particle_truth_after_sorting():
    hits = np.zeros((4, 5), dtype=np.float32)
    hits[:, 2] = [30, 10, 20, 40]
    truth = np.array([[3, 30, 30], [1, 10, 10], [2, 20, 20], [4, 40, 40]], dtype=np.float32)
    track = StrawTrack('event', 0, hits, np.array([3, 1, 2, 4]), np.arange(4), hits_truth=truth)
    batch = collate_fn([track])
    assert batch['coords_current'][0, :, 0].tolist() == [1, 2, 3]
    assert batch['coords_next'][0, :, 0].tolist() == [2, 3, 4]
    assert batch['targets'].tolist() == [[2, 3, 4]]


def test_evaluation_coordinate_stats_report_euclidean_and_axis_errors():
    stats = CoordinateStats()
    stats.update(np.array([[3., 4., 0.], [0., 0., 0.]]), np.zeros((2, 3)))
    result = stats.as_dict()
    assert result['support'] == 2
    assert result['mean_distance_mm'] == 2.5
    assert result['rmse_distance_mm'] == pytest.approx(np.sqrt(12.5))


def test_training_combines_coordinate_weights_and_uses_same_mask():
    module = StrawTrackNETModule(hidden_features=4, num_tubes=10, predict_coordinates=True,
                                coordinate_scale=(1., 1., 1.),
                                current_coordinate_loss_weight=2., next_coordinate_loss_weight=3.)
    module.log = lambda *args, **kwargs: None
    batch = {'coords_current': torch.zeros(1, 3, 3), 'coords_next': torch.zeros(1, 3, 3)}
    batch['coords_current'][:, 2] = float('nan')
    batch['coords_next'][:, 2] = float('nan')
    output = {'coords_t0': torch.ones(1, 2, 3), 'coords_t1': torch.full((1, 2, 3), 2.)}
    parts = {'loss': torch.tensor(5.)}
    module._add_coordinate_loss(parts, output, batch, torch.tensor([[True, False]]), 'train')
    torch.testing.assert_close(parts['loss'], torch.tensor(5. + 2. * 1. + 3. * 4.))


def test_training_fails_clearly_without_truth_coordinates():
    module = StrawTrackNETModule(hidden_features=4, num_tubes=10, predict_coordinates=True)
    with pytest.raises(ValueError, match='Regenerate'):
        module._add_coordinate_loss({'loss': torch.tensor(0.)}, {}, {}, torch.ones(1, 1, dtype=torch.bool), 'train')
