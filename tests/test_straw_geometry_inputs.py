import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from src.tracknet.data.dataset import PrebatchedStrawTracksDataset
from src.tracknet.loss import StrawTrackNetLoss
from src.tracknet.model import StrawTrackNET
from src.tracknet.training import StrawTrackNETModule


def test_straw_inputs_are_normalized_and_share_xyuv_embeddings():
    model = StrawTrackNET(
        input_features=5,
        hidden_features=4,
        num_tubes=5,
        plane_embedding_dim=3,
    )
    inputs = torch.tensor(
        [[[-750.0, -750.0, 0.0, 0.0, 1.0],
          [750.0, 750.0, 240.0, 5.0, 5.0]]]
    )

    encoded = model.encode_inputs(inputs)

    torch.testing.assert_close(encoded[0, 0, :4], -torch.ones(4))
    torch.testing.assert_close(encoded[0, 1, :4], torch.ones(4))
    torch.testing.assert_close(encoded[0, 0, 4:], encoded[0, 1, 4:])
    assert encoded.shape == (1, 2, 7)


def test_straw_model_rejects_legacy_lr_feature_in_new_schema():
    try:
        StrawTrackNET(input_features=6, use_plane_embedding=True)
    except ValueError as error:
        assert "x0, y0, z0, dr, station" in str(error)
    else:
        raise AssertionError("The new model unexpectedly accepted a sixth lr feature.")


def test_straw_loss_is_hard_cross_entropy_on_the_true_tube_only():
    logits = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]]])
    targets = torch.tensor([[1, 2]])
    mask = torch.tensor([[True, False]])
    loss = StrawTrackNetLoss()

    actual = loss({"tube_logits_t1": logits}, targets, mask)
    expected = F.cross_entropy(logits[:, :1].reshape(-1, 3), targets[:, :1].reshape(-1))

    torch.testing.assert_close(actual, expected)


def test_straw_ranking_loss_uses_hardest_negative_from_target_station():
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0, 100.0, -2.0, -3.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[1]])
    mask = torch.tensor([[True]])
    loss = StrawTrackNetLoss(
        ranking_loss_weight=0.25,
        ranking_margin=0.5,
        ranking_top_m=1,
        station_tube_counts=[3, 3],
    )

    parts = loss.loss_components(
        {"tube_logits_t1": logits}, targets, mask
    )

    # Class 2 is the hard negative. Class 3 has a much larger logit, but it is
    # in another station and must not affect the local ranking term.
    expected_ranking = torch.tensor(1.5)
    expected_ce = F.cross_entropy(logits.reshape(-1, 6), targets.reshape(-1))
    torch.testing.assert_close(parts["ranking_loss"], expected_ranking)
    torch.testing.assert_close(parts["hard_negative_gap"], torch.tensor(-1.0))
    torch.testing.assert_close(
        parts["loss"], expected_ce + 0.25 * expected_ranking
    )

    parts["ranking_loss"].backward()
    assert logits.grad[0, 0, 1] < 0
    assert logits.grad[0, 0, 2] > 0
    assert logits.grad[0, 0, 3] == 0


def test_straw_ranking_loss_is_zero_when_margin_is_satisfied():
    logits = torch.tensor([[[0.0, 2.0, 1.25, 20.0, 0.0]]])
    loss = StrawTrackNetLoss(
        ranking_loss_weight=1.0,
        ranking_margin=0.5,
        ranking_top_m=1,
        station_tube_counts=[3, 2],
    )

    parts = loss.loss_components(
        {"tube_logits_t1": logits},
        torch.tensor([[1]]),
        torch.tensor([[True]]),
    )

    torch.testing.assert_close(parts["ranking_loss"], torch.tensor(0.0))
    torch.testing.assert_close(parts["hard_negative_gap"], torch.tensor(0.75))


def test_straw_top_m_ranking_averages_three_strongest_negatives():
    logits = torch.tensor(
        [[[2.0, 3.0, 2.25, 1.75, 0.0]]],
        requires_grad=True,
    )
    loss = StrawTrackNetLoss(
        ranking_loss_weight=1.0,
        ranking_margin=0.5,
        ranking_top_m=3,
        station_tube_counts=[5],
    )

    parts = loss.loss_components(
        {"tube_logits_t1": logits},
        torch.tensor([[0]]),
        torch.tensor([[True]]),
    )

    torch.testing.assert_close(
        parts["ranking_loss"], torch.tensor((1.5 + 0.75 + 0.25) / 3)
    )
    torch.testing.assert_close(
        parts["top_m_negative_gap"], torch.tensor((-1.0 - 0.25 + 0.25) / 3)
    )
    parts["ranking_loss"].backward()
    assert logits.grad[0, 0, 0] < 0
    assert bool((logits.grad[0, 0, 1:4] > 0).all())
    assert logits.grad[0, 0, 4] == 0


def test_straw_plateau_scheduler_uses_configured_floor_and_threshold():
    module = StrawTrackNETModule(
        input_features=5,
        hidden_features=4,
        num_tubes=6,
        learning_rate=1e-4,
        lr_scheduler_factor=0.3,
        lr_scheduler_patience=5,
        lr_scheduler_threshold=1e-4,
        lr_scheduler_threshold_mode="abs",
        lr_scheduler_min_lr=1e-6,
    )

    optimizer_config = module.configure_optimizers()
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]

    assert scheduler.factor == 0.3
    assert scheduler.patience == 5
    assert scheduler.threshold == 1e-4
    assert scheduler.threshold_mode == "abs"
    assert scheduler.min_lrs == [1e-6]

    optimizer = optimizer_config["optimizer"]
    for _ in range(40):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == 1e-6


def test_training_starts_predictions_after_two_seed_hits():
    module = StrawTrackNETModule(
        input_features=5,
        hidden_features=4,
        num_tubes=4,
        seed_hits=2,
    )
    logits = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    batch = {
        "targets": torch.tensor([[1, 2, 3]]),
        "target_mask": torch.tensor([[True, True, True]]),
    }

    supervised, targets, mask = module._supervised_view(
        {"tube_logits_t1": logits}, batch
    )

    torch.testing.assert_close(supervised["tube_logits_t1"], logits[:, 1:])
    torch.testing.assert_close(targets, torch.tensor([[2, 3]]))
    torch.testing.assert_close(mask, torch.tensor([[True, True]]))


def test_prebatched_cache_rejects_legacy_input_columns():
    with tempfile.TemporaryDirectory() as temporary_dir:
        cache_dir = Path(temporary_dir)
        torch.save(
            {
                "num_tubes": 5,
                "input_columns": ["x0", "y0", "z0", "dr", "lr", "station"],
            },
            cache_dir / "metadata.pt",
        )

        try:
            PrebatchedStrawTracksDataset(
                cache_dir,
                batch_size=2,
                num_tubes=5,
                input_columns=["x0", "y0", "z0", "dr", "station"],
            )
        except ValueError as error:
            assert "Regenerate the cache" in str(error)
        else:
            raise AssertionError("A cache containing lr was unexpectedly accepted.")
