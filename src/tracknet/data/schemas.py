import numpy as np
import torch

from typing import TypedDict
from dataclasses import dataclass
from functools import cached_property


@dataclass
class Track:
    event_id: str
    track_id: int
    particle_id: int
    hits_xyz: np.ndarray
    px: np.float32
    py: np.float32
    pz: np.float32
    charge: np.int32
    volume_ids: np.ndarray
    layer_ids: np.ndarray
    module_ids: np.ndarray
    hit_ids: np.ndarray

    @cached_property
    def hits_cylindrical(self) -> np.ndarray:
        """
        Convert unnormalized hit coordinates from Cartesian (x, y, z) to cylindrical (r, phi, z).

        Returns:
            np.ndarray: A 2D array where each row contains the cylindrical coordinates
                        (r, phi, theta) corresponding to the Cartesian coordinates (x, y, z)
                        of each hit.
                        - r: Radial distance from beam line.
                        - phi: Azimuthal angle in the xy-plane.
                        - z: Position along beam direction.
        """
        x = self.hits_xyz[:, 0]
        y = self.hits_xyz[:, 1]
        z = self.hits_xyz[:, 2]

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return np.column_stack([r, phi, z])

    @cached_property
    def momentum_pt(self) -> float:
        """Transverse momentum pT = sqrt(px^2 + py^2)."""
        return np.sqrt(self.px**2 + self.py**2)

    def __post_init__(self):
        """
        Post-initialization method to sort hit data by radial distance.

        This method calculates the radial distance (r) of each hit from origin in spherical 
        coordinates using x, y, and z coordinates. It then sorts the hits and associated data
        (volume_ids, layer_ids, module_ids, hit_ids) based on this radial distance
        in ascending order.
        """
        r = np.sqrt(
            self.hits_xyz[:, 0]**2 +
            self.hits_xyz[:, 1]**2 +
            self.hits_xyz[:, 2]**2
        )
        sort_idx = np.argsort(r)
        self.hits_xyz = self.hits_xyz[sort_idx]
        self.volume_ids = self.volume_ids[sort_idx]
        self.layer_ids = self.layer_ids[sort_idx]
        self.module_ids = self.module_ids[sort_idx]
        self.hit_ids = self.hit_ids[sort_idx]
        _ = self.hits_cylindrical  # Calculate and cache the cylindrical coordinates

    def __repr__(self) -> str:
        """Detailed representation showing track properties and hit distribution."""
        # Count hits per volume-layer combination
        detector_info = []
        for volume, layer, module in zip(self.volume_ids, self.layer_ids, self.module_ids):
            # e.g. "vol7:[{'l2': 3, 'l4': 1}]"
            detector_info.append(
                f"vol{int(volume)}-l{int(layer)}-m{int(module)}"
            )

        hits_info = "\n\t\t".join(detector_info)

        return (f"Track(\n"
                f"\tevent_id={self.event_id},\n"
                f"\ttrack_id={self.track_id},\n"
                f"\tparticle_id={self.particle_id},\n"
                f"\thits={len(self.hits_xyz)},\n"
                f"\tpx={self.px:.2f}, py={self.py:.2f}, pz={self.pz:.2f},\n"
                f"\tpT={self.momentum_pt:.2f}, charge={self.charge},\n"
                f"\tdetector_info=[\n\t\t{hits_info}\n\t]\n"
                f")")

    def __str__(self) -> str:
        """Returns concise string representation focusing on key attributes."""
        return f"Track {self.track_id} with {len(self.hits_xyz)} hits, pT={self.momentum_pt:.2f}"

    def __len__(self) -> int:
        """Returns the number of hits in the track."""
        return len(self.hits_xyz)


@dataclass
class StrawTrack:
    event_id: str
    track_id: int
    hits_xyz: np.ndarray
    tube_ids: np.ndarray
    hit_ids: np.ndarray
    hits_wp: np.ndarray | None = None

    #@cached_property
    #def momentum_pt(self) -> float:
    #    """Transverse momentum pT = sqrt(px^2 + py^2)."""
    #    return np.sqrt(self.px**2 + self.py**2)

    def __post_init__(self):
        """
        Keep straw hits ordered along the detector depth when the feature vector contains
        z in the third column. The tube class labels and hit ids must follow the same order.
        """
        if self.hits_xyz.shape[0] == 0 or self.hits_xyz.shape[1] < 3:
            return

        sort_idx = np.argsort(self.hits_xyz[:, 2])
        self.hits_xyz = self.hits_xyz[sort_idx]
        self.tube_ids = self.tube_ids[sort_idx]
        self.hit_ids = self.hit_ids[sort_idx]
        if self.hits_wp is not None:
            self.hits_wp = self.hits_wp[sort_idx]

    def __repr__(self) -> str:
        """Detailed representation showing track properties and hit distribution."""

        return (f"Track(\n"
                f"\tevent_id={self.event_id},\n"
                f"\ttrack_id={self.track_id},\n"
                f"\thits={len(self.hits_xyz)},\n"
                #f"\tpx={self.px:.2f}, py={self.py:.2f}, pz={self.pz:.2f},\n"
                #f"\tpT={self.momentum_pt:.2f}, charge={self.charge},\n"
                f")")

    def __str__(self) -> str:
        """Returns concise string representation focusing on key attributes."""
        return f"Track {self.track_id} with {len(self.hits_xyz)} hits"#, pT={self.momentum_pt:.2f}"

    def __len__(self) -> int:
        """Returns the number of hits in the track."""
        return len(self.hits_xyz)


class BatchSample(TypedDict):
    inputs: torch.Tensor
    input_lengths: list[int]
    targets: torch.Tensor
    target_mask: torch.Tensor
