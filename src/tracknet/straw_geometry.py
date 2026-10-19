"""Tube-local coordinates for the fixed DriftSim V3 X/Y/U/V geometry."""

import math

import torch
from torch import nn


V3_TUBE_COUNTS = (151, 151, 213, 213, 151, 151, 213, 213)


def plane_frame(station):
    """Return unit normal n and wire direction u=(-n_y,n_x).

    Positive LR is the positive normal side, matching DriftSim V3's lr=+1.
    Padding must be masked before calling this function.
    """
    station = station.float()
    if not torch.isfinite(station).all() or torch.any(
        (station != station.round()) | (station < 1) | (station > 8)
    ):
        raise ValueError("Tube-local coordinates require integer V3 stations 1..8.")
    normals = station.new_tensor(
        [[1., 0.], [0., 1.], [1 / math.sqrt(2), 1 / math.sqrt(2)],
         [1 / math.sqrt(2), -1 / math.sqrt(2)]]
    )
    normal = normals[(station.long() - 1) % 4]
    direction = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
    return normal, direction


def tube_local_targets(truth, anchor, station):
    """Project particle truth relative to the wire anchor; no model inputs change."""
    normal, direction = plane_frame(station.float())
    displacement = truth[..., :2].float() - anchor[..., :2].float()
    signed_drift = (displacement * normal).sum(dim=-1, keepdim=True)
    along = (displacement * direction).sum(dim=-1, keepdim=True)
    return along, (signed_drift >= 0).float(), signed_drift.abs()


def reconstruct_tube_position(along, lr_logits, measurement, *, soft=False):
    """Combine a local prediction with an OBSERVED [x0,y0,z0,dr,station].

    Future dr is not predicted. For t+1 the caller must supply the measurement
    of the candidate tube, after candidate selection. soft=True returns the
    posterior mean between the two LR branches rather than a physical branch.
    """
    if measurement.shape[-1] != 5:
        raise ValueError("Tube measurements must contain [x0,y0,z0,dr,station].")
    if along.shape != (*measurement.shape[:-1], 1) or lr_logits.shape != along.shape:
        raise ValueError("Along and LR predictions must have shape (..., 1).")
    if not torch.isfinite(measurement).all() or torch.any(measurement[..., 3] < 0):
        raise ValueError("Tube measurements must be finite with non-negative dr.")
    if not torch.isfinite(along).all() or not torch.isfinite(lr_logits).all():
        raise ValueError("Local position and LR logits must be finite.")
    normal, direction = plane_frame(measurement[..., 4].float())
    sign = 2 * lr_logits.float().sigmoid() - 1 if soft else torch.where(lr_logits >= 0, 1., -1.)
    xy = (measurement[..., :2].float() + along.float() * direction
          + sign * measurement[..., 3:4].float() * normal)
    return torch.cat((xy, measurement[..., 2:3].float()), dim=-1)


class V3TubeGeometry(nn.Module):
    """Lookup of anchors/stations from dense V3 class ids, used for targets."""

    def __init__(self, num_tubes, station_tube_counts=None):
        super().__init__()
        if num_tubes != sum(V3_TUBE_COUNTS) or (
            station_tube_counts is not None and tuple(station_tube_counts) != V3_TUBE_COUNTS
        ):
            raise ValueError("Tube-relative heads require the fixed 1456-class DriftSim V3 geometry.")
        stations = torch.repeat_interleave(torch.arange(1, 9), torch.tensor(V3_TUBE_COUNTS)).float()
        normal, _ = plane_frame(stations)
        local_ids = torch.cat([torch.arange(count) for count in V3_TUBE_COUNTS]).float()
        minimum = -750. * normal.abs().sum(dim=-1)
        xy = normal * (minimum + 10. * local_ids).unsqueeze(-1)
        z = (stations - 1) * (240. / 7)
        # Geometry is deterministic and is reconstructed when loading checkpoints.
        self.register_buffer("measurements", torch.cat((xy, z[:, None], torch.zeros_like(z[:, None]), stations[:, None]), -1), persistent=False)

    def for_classes(self, classes):
        if not torch.isfinite(classes).all() or torch.any(
            (classes < 0) | (classes >= len(self.measurements)) | (classes != classes.long())
        ):
            raise ValueError("Tube class is outside the V3 geometry.")
        return self.measurements[classes.long()]
