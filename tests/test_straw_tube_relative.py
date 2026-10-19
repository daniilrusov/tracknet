"""Geometry, causal supervision, and checkpoint checks for along-wire/LR heads."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from driftsim_v3 import PLANE_NORMALS
from evaluate_straw_checkpoint import load_model, TubeLocalStats
from src.tracknet.loss import StrawTubeLocalLoss
from src.tracknet.metrics import StrawLRAccuracy
from src.tracknet.model import StrawTrackNET
from src.tracknet.straw_geometry import (
    V3TubeGeometry, plane_frame, tube_local_targets, reconstruct_tube_position,
)
from src.tracknet.training import StrawTrackNETModule


def measurements():
    geometry = V3TubeGeometry(1456)
    classes = torch.tensor([70, 220, 400, 600, 800, 950, 1100, 1300])
    result = geometry.for_classes(classes).clone()
    result[:, 3] = torch.arange(1., 9.) / 2
    return result


def test_v3_frames_match_generator_and_local_roundtrip_on_every_plane():
    measured = measurements()
    along = torch.arange(-4., 4.)[:, None] * 40
    lr_logits = torch.tensor([-1., 1.] * 4)[:, None] * 20
    normal, direction = plane_frame(measured[:, 4])
    expected_normals = np.stack([PLANE_NORMALS[plane] for plane in ('X', 'Y', 'U', 'V') * 2])
    np.testing.assert_allclose(normal.numpy(), expected_normals, atol=1e-7)
    torch.testing.assert_close((normal * direction).sum(-1), torch.zeros(8))
    xyz = reconstruct_tube_position(along, lr_logits, measured)
    target_along, target_lr, target_dr = tube_local_targets(xyz, measured[:, :3], measured[:, 4])
    torch.testing.assert_close(target_along, along, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(target_lr, (lr_logits > 0).float())
    torch.testing.assert_close(target_dr, measured[:, 3:4], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(xyz[:, 2], measured[:, 2])
    mean = reconstruct_tube_position(along, torch.zeros_like(lr_logits), measured, soft=True)
    torch.testing.assert_close(((mean[:, :2] - measured[:, :2]) * normal).sum(-1), torch.zeros(8), atol=1e-4, rtol=0)


def test_relative_loss_masks_nan_padding_and_backpropagates_both_tasks():
    along = torch.tensor([[[float('nan')], [6.], [float('nan')]]], requires_grad=True)
    logits = torch.tensor([[[float('nan')], [0.], [float('nan')]]], requires_grad=True)
    target_along = torch.tensor([[[float('nan')], [3.], [float('nan')]]])
    lr = torch.tensor([[[float('nan')], [1.], [float('nan')]]])
    mask = torch.tensor([[False, True, False]])
    parts = StrawTubeLocalLoss(along_scale=3., lr_weight=2.)(along, logits, target_along, lr, mask)
    torch.testing.assert_close(parts['loss'], torch.tensor(1. + 2 * np.log(2), dtype=torch.float32))
    parts['loss'].backward()
    for grad in (along.grad, logits.grad):
        assert torch.isfinite(grad).all()
        assert grad[0, 0] == 0 and grad[0, 2] == 0 and grad[0, 1] != 0


def test_relative_heads_are_causal_and_all_parts_receive_gradients():
    torch.manual_seed(42)
    module = StrawTrackNETModule(hidden_features=8, predict_coordinates=True,
                                coordinate_representation='tube_relative')
    module.log = lambda *args, **kwargs: None
    measured = measurements()[:5]
    along = torch.arange(5.)[:, None] * 20
    xyz = reconstruct_tube_position(along, torch.tensor([[-5.], [5.], [-5.], [5.], [-5.]]), measured)
    # A five-hit track; the last observed hit is only the future supervision target.
    batch = {'inputs': measured[None, :-1], 'input_lengths': [4],
             'targets': torch.tensor([[220, 400, 600, 800]]),
             'target_mask': torch.tensor([[False, True, True, True]]),
             'coords_current': xyz[None, :-1], 'coords_next': xyz[None, 1:]}
    output = module(batch)
    changed = batch['inputs'].clone()
    changed[:, 2:, :4] += 100
    altered = module.model(changed, [4])
    assert set(output) == {'tube_logits_t1', 'along_t0', 'along_t1', 'lr_logits_t0', 'lr_logits_t1'}
    for key in output:
        torch.testing.assert_close(output[key][:, :2], altered[key][:, :2])
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    for part in (module.model.rnn, module.model.relative_current,
                 module.model.relative_next, module.model.tube_classifier):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in part.parameters())
        assert sum(float(parameter.grad.abs().sum()) for parameter in part.parameters()) > 0


def test_training_relative_targets_use_current_input_and_true_future_tube_with_same_mask():
    module = StrawTrackNETModule(hidden_features=4, predict_coordinates=True,
                                coordinate_representation='tube_relative')
    module.log = lambda *args, **kwargs: None
    measured = measurements()
    current, future = measured[1], measured[2]
    current_truth = reconstruct_tube_position(torch.tensor([[12.]]), torch.tensor([[-20.]]), current[None])
    future_truth = reconstruct_tube_position(torch.tensor([[-32.]]), torch.tensor([[20.]]), future[None])
    batch = {'inputs': torch.zeros(1, 3, 5), 'targets': torch.tensor([[0, 400, 0]]),
             'coords_current': torch.full((1, 3, 3), float('nan')),
             'coords_next': torch.full((1, 3, 3), float('nan'))}
    batch['inputs'][0, 1] = current
    batch['coords_current'][0, 1] = current_truth[0]
    batch['coords_next'][0, 1] = future_truth[0]
    output = {'along_t0': torch.tensor([[[12.], [float('nan')]]]),
              'along_t1': torch.tensor([[[-32.], [float('nan')]]]),
              'lr_logits_t0': torch.tensor([[[-20.], [float('nan')]]]),
              'lr_logits_t1': torch.tensor([[[20.], [float('nan')]]])}
    parts = {'loss': torch.tensor(0.)}
    module._add_coordinate_loss(parts, output, batch, torch.tensor([[True, False]]), 'train')
    assert parts['loss'] < 1e-6
    assert module.train_current_lr_accuracy.compute() == 1
    assert module.train_next_lr_accuracy.compute() == 1
    assert module.train_current_along_rmse.compute() < 1e-4
    assert module.train_next_along_rmse.compute() < 1e-4


def test_relative_checkpoint_roundtrip_and_local_statistics(tmp_path):
    module = StrawTrackNETModule(hidden_features=4, predict_coordinates=True,
                                coordinate_representation='tube_relative', along_coordinate_scale=100.)
    path = tmp_path / 'model.ckpt'
    torch.save({'state_dict': module.state_dict(), 'hyper_parameters': dict(module.hparams)}, path)
    model, _ = load_model(path, torch.device('cpu'))
    assert model.coordinate_representation == 'tube_relative'
    assert model.along_coordinate_scale == 100.
    for key, value in module.model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
    measured = measurements()
    along = torch.zeros(8, 1)
    logits = torch.full((8, 1), 20.)
    truth = reconstruct_tube_position(along, logits, measured)
    stats = TubeLocalStats()
    stats.update(along, logits, truth, measured)
    result = stats.as_dict()
    assert result['support'] == 8 and result['lr_accuracy'] == 1
    assert result['along_rmse_mm'] < 1e-4
    assert result['xyz_hard_given_measurement']['rmse_distance_mm'] == 0


def test_lr_metric_counts_hits_across_batches_and_resets():
    metric = StrawLRAccuracy()
    metric.update(torch.tensor([[[1.], [-1.]]]), torch.tensor([[[1.], [1.]]]), torch.ones(1, 2, dtype=torch.bool))
    metric.update(torch.tensor([[[0.]]]), torch.tensor([[[1.]]]), torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(metric.compute(), torch.tensor(2 / 3))
    metric.reset()
    assert metric.count == 0


@pytest.mark.parametrize('kwargs', [dict(num_tubes=1208), dict(input_features=6, use_plane_embedding=False)])
def test_relative_geometry_rejects_incompatible_legacy_settings(kwargs):
    with pytest.raises(ValueError, match='V3'):
        StrawTrackNET(predict_coordinates=True, coordinate_representation='tube_relative', **kwargs)


def test_relative_heads_keep_cache_padding_when_all_tracks_are_short():
    model = StrawTrackNET(hidden_features=4, num_tubes=1456, predict_coordinates=True,
                          coordinate_representation='tube_relative')
    inputs = torch.zeros(1, 7, 5)
    inputs[0, :2] = measurements()[:2]
    output = model(inputs, [2])
    assert all(value.shape[:2] == (1, 7) for value in output.values())
