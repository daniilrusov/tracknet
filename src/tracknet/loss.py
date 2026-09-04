import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import TrackPrediction, PointPrediction, StrawTubePrediction


class PointInAreaLoss(nn.Module):
    """
    Computes the normalized distance between predicted and true hit positions.

    The loss evaluates how well predicted coordinates match true hit positions, normalized 
    by predicted search radius:

    PointInAreaLoss = sqrt(
        ( (x_pred - x_true)^2 + (y_pred - y_true)^2 + (z_pred - z_true)^2 ) / 3 * R_pred^2
    )

    where (x,y,z)_pred are predicted coordinates and R_pred is the predicted search radius.

    Returns:
        torch.Tensor: Concatenated loss values for t1 and t2 predictions with shape 
            (batch_size, 2*seq_len-1).
    """

    def __init__(self):
        super(PointInAreaLoss, self).__init__()

    def forward(self, preds: TrackPrediction, target: torch.Tensor):
        if preds['coords_t1'].size(0) != target.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                             'the prediction and target must be equal. '
                             f'{preds["coords_t1"].size(0) != target.size(0)}')

        if target.shape[-1] < 3:
            raise ValueError('Target must be 3-dimensional (x, y, z), '
                             f'but got target.shape[2] = {target.size(2)}')

        t1_coords_diff = preds['coords_t1'] - target
        # for the last hit, we don't have the next hit
        # that's why we exclude the last prediction from loss
        # we start from the second hit according to StepAhead TrackNET procedure
        t2_coords_diff = preds['coords_t2'][:, :-1] - target[:, 1:]
        t1_loss = t1_coords_diff / preds['radius_t1']
        # exclude the last prediction from loss
        t2_loss = t2_coords_diff / preds['radius_t2'][:, :-1]
        # equal to L2 norm, sqrt(sum(x_i^2))
        t1_loss = torch.norm(t1_loss, dim=-1)
        t2_loss = torch.norm(t2_loss, dim=-1)
        return torch.cat((t1_loss, t2_loss), dim=1)


class AreaSizeLoss(nn.Module):
    """
    Penalizes large search areas to prevent trivial solutions.

    The loss is simply the square of predicted search radius:
    AreaSizeLoss = R_pred^2

    This term prevents the model from predicting arbitrarily large search regions
    that would trivially contain the true hits.

    Returns:
        torch.Tensor: Concatenated squared radii for t1 and t2 predictions with shape 
            (batch_size, 2*seq_len-1).
    """

    def __init__(self):
        super(AreaSizeLoss, self).__init__()

    def forward(self, preds: TrackPrediction) -> torch.Tensor:
        r1_loss = torch.pow(preds['radius_t1'][:, :, 0], 2)
        # for the last hit, we don't have the next hit
        # so we need to exclude the last prediction from loss
        r2_loss = torch.pow(preds['radius_t2'][:, :-1, 0], 2)
        return torch.cat((r1_loss, r2_loss), dim=1)


class TrackNetLoss(nn.Module):
    """
    Combined loss function for TrackNET training.

    Balances between hit position accuracy and search area size:
    TrackNETLoss = α * PointInAreaLoss + (1-α) * AreaSizeLoss

    where:
    - α controls the trade-off between position accuracy and search area size
    - PointInAreaLoss measures normalized distance between predicted and true positions
    - AreaSizeLoss penalizes large search areas

    Args:
        alpha (float, optional): Weight factor in [0,1]. Higher values prioritize 
            position accuracy over small search areas. Defaults to 0.9.

    Returns:
        torch.Tensor: Scalar loss value averaged over masked valid positions.
    """

    def __init__(self, alpha=0.9):
        super(TrackNetLoss, self).__init__()

        if alpha > 1 or alpha < 0:
            raise ValueError('Weighting factor alpha must be in range [0, 1], '
                             f'but got alpha={alpha}')
        self.alpha = alpha
        self.point_in_area_loss = PointInAreaLoss()
        self.area_size_loss = AreaSizeLoss()

    def forward(
        self,
        preds: TrackPrediction,
        targets: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        points_in_area = self.point_in_area_loss(preds, targets)
        area_size = self.area_size_loss(preds)
        loss = self.alpha * points_in_area + \
            (1 - self.alpha) * area_size
        return loss.masked_select(target_mask).mean().float()


class StrawPointInAreaLoss(nn.Module):
    """
    Computes the normalized distance between predicted and true hit positions.

    The loss evaluates how well predicted coordinates match true hit positions, normalized 
    by predicted search radius:

    PointInAreaLoss = sqrt(
        ( (x_pred - x_true)^2 + (y_pred - y_true)^2 + (z_pred - z_true)^2 ) / 3 * R_pred^2
    )

    where (x,y,z)_pred are predicted coordinates and R_pred is the predicted search radius.

    Returns:
        torch.Tensor: Concatenated loss values for t1 and t2 predictions with shape 
            (batch_size, 2*seq_len-1).
    """

    def __init__(self):
        super(StrawPointInAreaLoss, self).__init__()

    def forward(self, preds: PointPrediction, target: torch.Tensor):
        if preds['coords_t1'].size(0) != target.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                             'the prediction and target must be equal. '
                             f'{preds["coords_t1"].size(0) != target.size(0)}')

        if target.shape[-1] < 3:
            raise ValueError('Target must be 3-dimensional (x, y, z), '
                             f'but got target.shape[2] = {target.size(2)}')

        t1_coords_diff = preds['coords_t1'] - target

        t1_loss = t1_coords_diff

        # equal to L2 norm, sqrt(sum(x_i^2))
        t1_loss = torch.norm(t1_loss, dim=-1)

        return t1_loss


class StrawTrackNetLoss(nn.Module):
    """Cross-entropy with an optional same-station top-M ranking margin.

    The ranking term directly optimizes the ordering that determines top-1:
    the true tube logit must exceed each of the M strongest incorrect logits
    from the target station by ``ranking_margin``. The hinge penalties are
    averaged, so changing M does not multiply the loss scale. Cross-entropy
    remains the primary objective and continues to train the full 1456-class
    distribution.
    """

    def __init__(
        self,
        ranking_loss_weight: float = 0.0,
        ranking_margin: float = 0.5,
        ranking_top_m: int = 1,
        station_tube_counts: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if ranking_loss_weight < 0:
            raise ValueError("ranking_loss_weight must be non-negative.")
        if ranking_margin < 0:
            raise ValueError("ranking_margin must be non-negative.")
        if ranking_top_m < 1:
            raise ValueError("ranking_top_m must be at least one.")
        if station_tube_counts is not None:
            station_tube_counts = tuple(int(count) for count in station_tube_counts)
            if not station_tube_counts or any(count < 2 for count in station_tube_counts):
                raise ValueError(
                    "station_tube_counts must contain at least two tubes per station."
                )
            if ranking_top_m >= min(station_tube_counts):
                raise ValueError(
                    "ranking_top_m must be smaller than every station tube count."
                )
        if ranking_loss_weight > 0 and station_tube_counts is None:
            raise ValueError(
                "station_tube_counts is required when hard-negative ranking is enabled."
            )

        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_top_m = int(ranking_top_m)
        self.station_tube_counts = station_tube_counts

    def loss_components(
        self,
        preds: StrawTubePrediction,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the total loss and its independently loggable components."""
        logits = preds['tube_logits_t1']
        if logits.size(0) != targets.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                             'the prediction and target must be equal. '
                             f'{logits.size(0) != targets.size(0)}')

        if targets.dim() != 2:
            raise ValueError('Straw tube targets must have shape '
                             f'(batch_size, seq_len), got {tuple(targets.shape)}')

        if logits.shape[:2] != targets.shape or target_mask.shape != targets.shape:
            raise ValueError(
                "Logit sequence, targets and target_mask must share batch/step dimensions."
            )

        if not bool(target_mask.any()):
            raise ValueError("target_mask does not contain any supervised predictions.")

        flat_logits = logits[target_mask].float()
        flat_targets = targets[target_mask]
        cross_entropy = F.cross_entropy(flat_logits, flat_targets)

        if self.ranking_loss_weight == 0:
            zero = cross_entropy.new_zeros(())
            return {
                "loss": cross_entropy,
                "cross_entropy": cross_entropy,
                "ranking_loss": zero,
                "hard_negative_gap": zero,
                "top_m_negative_gap": zero,
            }

        assert self.station_tube_counts is not None
        if sum(self.station_tube_counts) != flat_logits.size(1):
            raise ValueError(
                "station_tube_counts sum to "
                f"{sum(self.station_tube_counts)}, but logits contain "
                f"{flat_logits.size(1)} tube classes."
            )

        true_logits = flat_logits.gather(1, flat_targets[:, None]).squeeze(1)
        ranking_loss_sum = flat_logits.new_zeros(())
        hard_negative_gap_sum = flat_logits.new_zeros(())
        top_m_negative_gap_sum = flat_logits.new_zeros(())
        assigned_count = 0
        start = 0
        for count in self.station_tube_counts:
            end = start + count
            rows = (flat_targets >= start) & (flat_targets < end)
            if bool(rows.any()):
                station_logits = flat_logits[rows, start:end].clone()
                local_targets = flat_targets[rows] - start
                station_logits.scatter_(1, local_targets[:, None], float("-inf"))
                top_negative_logits = station_logits.topk(
                    self.ranking_top_m, dim=1
                ).values
                gaps = true_logits[rows, None] - top_negative_logits
                ranking_loss_sum = ranking_loss_sum + F.relu(
                    self.ranking_margin - gaps
                ).sum()
                hard_negative_gap_sum = hard_negative_gap_sum + gaps[:, 0].sum()
                top_m_negative_gap_sum = top_m_negative_gap_sum + gaps.sum()
                assigned_count += int(rows.sum())
            start = end

        if assigned_count != len(flat_targets):
            configured_classes = sum(self.station_tube_counts)
            bad_targets = flat_targets[
                (flat_targets < 0) | (flat_targets >= configured_classes)
            ]
            bad_target = int(bad_targets[0].detach().cpu())
            raise ValueError(
                f"Target class {bad_target} is outside configured station ranges."
            )

        ranking_loss = ranking_loss_sum / (assigned_count * self.ranking_top_m)
        hard_negative_gap = hard_negative_gap_sum / assigned_count
        top_m_negative_gap = top_m_negative_gap_sum / (
            assigned_count * self.ranking_top_m
        )
        total_loss = cross_entropy + self.ranking_loss_weight * ranking_loss
        return {
            "loss": total_loss,
            "cross_entropy": cross_entropy,
            "ranking_loss": ranking_loss,
            "hard_negative_gap": hard_negative_gap,
            "top_m_negative_gap": top_m_negative_gap,
        }

    def forward(
        self,
        preds: StrawTubePrediction,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_components(preds, targets, target_mask)["loss"]
