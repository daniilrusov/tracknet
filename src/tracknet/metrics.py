import torch
import numpy as np
from torchmetrics import Metric
from scipy.spatial import cKDTree
from typing import Optional
from .data.transformations import MinMaxNormalizeXYZ


class SearchAreaMetric(Metric):
    """
    Calculates the average search area of predicted spherical regions.

    This metric computes the mean volume of spherical search regions predicted by 
    TrackNET for either t1 or t2 time steps. For t2 predictions, the last position 
    is excluded since there is no next hit to predict.

    The search area is calculated as 4/3 * pi * r^3 for each prediction, where r is 
    the predicted sphere radius. Only valid positions according to the target mask 
    are included in the average.

    Args:
        time_step (str): Which predictions to evaluate - either 't1' or 't2'.
            't1' evaluates predictions for immediate next hits
            't2' evaluates predictions for hits after next

    Example:
        >>> metric = SearchAreaMetric('t1')
        >>> preds = {
        ...     'radius_t1': torch.tensor([[[1.0]], [[2.0]]]),  # batch_size=2, seq_len=1
        ...     'radius_t2': torch.tensor([[[1.5]], [[2.5]]])
        ... }
        >>> target_mask = torch.tensor([[True], [True]])  # batch_size=2, seq_len=1
        >>> metric.update(preds, target_mask)
        >>> metric.compute()
        tensor(18.8496)  # mean volume of spheres with r=1.0 and r=2.0

    Note:
        The metric automatically handles distributed training by summing total_area
        and total_count across all processes before computing the final average.
    """

    def __init__(self, time_step='t1'):
        super().__init__()
        if time_step not in ['t1', 't2']:
            raise ValueError("time_step must be 't1' or 't2'")
        self.time_step = time_step
        self.add_state(
            "total_area",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_count",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: dict, target_mask: torch.Tensor):
        radius = preds[f'radius_{self.time_step}']

        # For t2, exclude last prediction as there's no next hit
        if self.time_step == 't2':
            radius = radius[:, :-1]

        # Calculate areas of spheres (4/3 * pi * r^3)
        areas = (4/3) * torch.pi * torch.pow(radius[..., 0], 3)

        # Apply mask for valid predictions
        if self.time_step == 't2':
            mask = target_mask[:, -radius.size(1):]
        else:
            mask = target_mask[:, :radius.size(1)]

        valid_areas = areas[mask]

        self.total_area += valid_areas.sum()
        self.total_count += valid_areas.numel()

    def compute(self):
        return self.total_area / self.total_count


class HitEfficiencyMetric(Metric):
    """
    Calculates the fraction of true hits that lie inside predicted search regions.

    For each prediction, checks if the actual hit position falls within the
    predicted spherical search region (i.e. if the distance between predicted and
    actual position is less than or equal to the predicted radius). The metric
    returns the fraction of hits that satisfy this criterion.

    Args:
        time_step (str): Which predictions to evaluate - either 't1' or 't2'.
            't1' evaluates predictions for immediate next hits
            't2' evaluates predictions for hits after next

    Example:
        >>> metric = HitEfficiencyMetric('t1')
        >>> preds = {
        ...     'coords_t1': torch.tensor([[[1.0, 0.0, 0.0]]]),  # predicted position
        ...     'radius_t1': torch.tensor([[[1.5]]]),  # search radius
        ... }
        >>> targets = torch.tensor([[[2.0, 0.0, 0.0]]])  # true position
        >>> target_mask = torch.tensor([[True]])
        >>> metric.update(preds, targets, target_mask)
        >>> metric.compute()
        tensor(1.0)  # hit is within radius 1.5 of prediction

    Note:
        - For t2 predictions, compares with hits at t+2 and excludes the last position
        - The metric automatically handles distributed training by summing hits_inside
          and total_hits across all processes
    """

    def __init__(self, time_step='t1'):
        super().__init__()
        if time_step not in ['t1', 't2']:
            raise ValueError("time_step must be 't1' or 't2'")
        self.time_step = time_step
        self.add_state(
            "hits_inside",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_hits",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: dict, targets: torch.Tensor, target_mask: torch.Tensor):
        coords = preds[f'coords_{self.time_step}']
        radius = preds[f'radius_{self.time_step}']

        # For t2, we compare with next hits and exclude last prediction
        if self.time_step == 't2':
            coords = coords[:, :-1]
            radius = radius[:, :-1]
            targets = targets[:, 1:]

        # Calculate distances between predicted and true positions
        distances = torch.norm(coords - targets, dim=-1)

        # Check if hits are inside spheres (distance <= radius)
        hits_inside = distances <= radius[..., 0]

        # Apply mask for valid predictions
        if self.time_step == 't2':
            mask = target_mask[:, -coords.size(1):]
        else:
            mask = target_mask[:, :coords.size(1)]

        valid_hits = hits_inside[mask]

        self.hits_inside += valid_hits.sum()
        self.total_hits += valid_hits.numel()

    def compute(self):
        return self.hits_inside.float() / self.total_hits


class HitDensityMetric(Metric):
    """
    Calculates the average number of hits expected in predicted search regions,
    accounting for coordinate normalization during training.

    Args:
        density_stats_path (str): Path to .npz file with density statistics
        time_step (str): Which predictions to evaluate - either 't1' or 't2'
        normalizer (MinMaxNormalizeXYZ): Normalizer instance used in training
            for denormalizing coordinates
    """

    def __init__(
        self,
        density_stats_path: str,
        time_step: str = 't1',
        normalizer: Optional[MinMaxNormalizeXYZ] = None
    ):
        super().__init__()

        if time_step not in ['t1', 't2']:
            raise ValueError("time_step must be 't1' or 't2'")

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            # If no normalizer provided, use identity transform (no normalization)
            self.normalizer = MinMaxNormalizeXYZ.identity()

        self.time_step = time_step

        try:
            density_data = np.load(density_stats_path, allow_pickle=True)
            required_keys = ['density_map', 'voxel_centers', 'grid_info']
            if not all(key in density_data for key in required_keys):
                raise KeyError("Missing required data in density stats file")

            self.density_map = density_data['density_map']
            self.voxel_centers = density_data['voxel_centers']
            self.occupied_mask = density_data['occupied_mask']
            self.grid_info = density_data['grid_info'].item()

            if len(self.density_map) == 0 or len(self.voxel_centers) == 0:
                raise ValueError("Empty density map or voxel centers")

            # Filter to keep only active voxels
            self.active_voxel_centers = self.voxel_centers[self.occupied_mask]
            self.active_densities = self.density_map[self.occupied_mask]

        except Exception as e:
            raise ValueError(
                f"Failed to load density statistics from {density_stats_path}: {e}")

        self.kdtree = cKDTree(self.active_voxel_centers)

        self.add_state(
            "total_density",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_count",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def estimate_hits_in_sphere(
        self,
        center: np.ndarray,
        radius: float,
        max_samples: int = 100
    ) -> float:
        """
        Estimate number of hits in a spherical region using density map.

        Args:
            center: Coordinates in detector space (mm)
            radius: Sphere radius in detector space (mm)
            max_samples: Maximum number of voxels to sample for density estimation

        Returns:
            Estimated number of hits in the sphere
        """
        if radius <= 0:
            return 0.0

        indices = self.kdtree.query_ball_point(center, radius)

        if not indices:
            return 0.0

        if len(indices) > max_samples:
            indices = np.random.choice(indices, max_samples, replace=False)

        densities = self.active_densities[indices]

        active_voxels_volume = len(indices) * self.grid_info['voxel_size']**3
        voxel_volume = self.grid_info['voxel_size']**3
        sphere_volume = (4/3) * np.pi * radius**3

        # Use the minimum of sphere volume and active voxels volume to avoid overestimation
        effective_volume = min(sphere_volume, active_voxels_volume)

        # Calculate hits using the effective volume
        estimated_hits = np.mean(densities) * effective_volume / voxel_volume

        return float(estimated_hits)

    def update(self, preds: dict, target_mask: torch.Tensor):
        """
        Update metric states with new predictions.

        Args:
            preds: Dictionary with model predictions in normalized space
            target_mask: Boolean mask for valid predictions
        """
        coords = preds[f'coords_{self.time_step}']
        radius = preds[f'radius_{self.time_step}']

        if self.time_step == 't2':
            coords = coords[:, :-1]
            radius = radius[:, :-1]

        # Denormalize coordinates and radii to detector space
        coords_detector = self.normalizer.denormalize_xyz(coords)

        # Scale radii by average coordinate range to match detector space
        coords_range = torch.from_numpy(
            self.normalizer.range_xyz).to(radius.device)
        avg_scale = coords_range.mean()
        radius_detector = radius * avg_scale

        coords_np = coords_detector.detach().cpu().numpy()
        radius_np = radius_detector[..., 0].detach().cpu().numpy()

        if self.time_step == 't2':
            mask = target_mask[:, -coords.size(1):]
        else:
            mask = target_mask[:, :coords.size(1)]

        mask_np = mask.detach().cpu().numpy()

        for i in range(coords_np.shape[0]):
            for j in range(coords_np.shape[1]):
                if mask_np[i, j]:
                    estimated_hits = self.estimate_hits_in_sphere(
                        coords_np[i, j],
                        radius_np[i, j]
                    )
                    self.total_density += torch.tensor(estimated_hits)
                    self.total_count += 1

    def compute(self):
        """Compute average estimated hits per search region."""
        if self.total_count > 0:
            return self.total_density.float() / self.total_count
        return torch.tensor(0.0)


class StrawHitEfficiencyMetric(Metric):
    """Accuracy of next straw tube classification."""

    def __init__(self, time_step='t1'):
        super().__init__()
        if time_step not in ['t1', 't2']:
            raise ValueError("time_step must be 't1' or 't2'")
        self.time_step = time_step
        self.add_state(
            "correct_hits",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_hits",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: dict, targets: torch.Tensor, target_mask: torch.Tensor):
        logits = preds['tube_logits_t1']
        predicted_tubes = torch.argmax(logits, dim=-1)

        # Apply mask for valid predictions
        mask = target_mask[:, :predicted_tubes.size(1)]

        valid_hits = predicted_tubes[mask] == targets[:, :predicted_tubes.size(1)][mask]

        self.correct_hits += valid_hits.sum()
        self.total_hits += valid_hits.numel()

    def compute(self):
        if self.total_hits == 0:
            return torch.tensor(0.0, device=self.total_hits.device)
        return self.correct_hits.float() / self.total_hits
