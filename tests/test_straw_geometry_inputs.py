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
