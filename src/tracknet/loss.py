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
    """Hard cross-entropy for the single true next-tube class."""

    def forward(
        self,
        preds: StrawTubePrediction,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
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

        return F.cross_entropy(
            logits[target_mask].float(),
            targets[target_mask],
        )
