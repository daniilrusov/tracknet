from abc import ABC, abstractmethod
from typing import Optional
from .schemas import Track


class TrackFilter(ABC):
    """Base class for all track filters."""

    @abstractmethod
    def __call__(self, track: Track) -> bool:
        """Return True if track should be kept, False if filtered out."""
        pass


class MinHitsFilter(TrackFilter):
    """
    Filter tracks based on the minimum number of hits.

    Args:
        min_hits (int): Minimum number of hits required for a track to be kept. Must be greater than 3.

    Methods:
        __call__(track: Track) -> bool:
            Checks if the track has at least the minimum number of hits.
            Returns True if the track should be kept, False otherwise.
    """

    def __init__(self, min_hits: int = 3):
        if min_hits <= 2:
            raise ValueError("min_hits must be greater than 2")
        self.min_hits = min_hits

    def __call__(self, track: Track) -> bool:
        return len(track.hits_xyz) >= self.min_hits


class MaxHitsFilter(TrackFilter):
    """
    Filter tracks based on the minimum number of hits.

    Args:
        min_hits (int): Minimum number of hits required for a track to be kept. Must be greater than 3.

    Methods:
        __call__(track: Track) -> bool:
            Checks if the track has at least the minimum number of hits.
            Returns True if the track should be kept, False otherwise.
    """

    def __init__(self, max_hits: int = 100):
        if max_hits <= 2:
            raise ValueError("max_hits must be greater than 2")
        self.max_hits = max_hits

    def __call__(self, track: Track) -> bool:
        return len(track.hits_xyz) <= self.max_hits


class FirstLayerFilter(TrackFilter):
    """
    Filter tracks that don't start from specified detector layers.

    Args:
        valid_first_layers (set[tuple[int, int]]): Set of valid (volume_id, layer_id) 
            combinations for the first hit.

    Methods:
        __call__(track: Track) -> bool:
            Checks if the first hit of the track is in the valid_first_layers set.
            Returns True if the track should be kept, False otherwise.
    """

    def __init__(self, valid_first_layers: set[tuple[int, int]]):
        self.valid_first_layers = valid_first_layers

    def __call__(self, track: Track) -> bool:
        if len(track.hits_xyz) == 0:
            return False
        first_hit = (int(track.volume_ids[0]), int(track.layer_ids[0]))
        return first_hit in self.valid_first_layers


class PtFilter(TrackFilter):
    """
    Filter tracks based on transverse momentum (pT).

    Args:
        min_pt (float): Minimum transverse momentum required for a track to be kept.

    Methods:
        __call__(track: Track) -> bool:
            Checks if the track's transverse momentum is greater than or equal to the minimum pT.
            Returns True if the track should be kept, False otherwise.
    """

    def __init__(self, min_pt: float):
        self.min_pt = min_pt

    def __call__(self, track: Track) -> bool:
        return track.momentum_pt >= self.min_pt


class FilterPipeline:
    """
    Manages a sequence of track filters.

    Args:
        filters (Optional[list[TrackFilter]]): List of TrackFilter instances to apply to each track.
                                               If None, an empty list is used.

    Methods:
        __call__(track: Track) -> bool:
            Apply all filters in sequence to the given track.
            Returns True if the track passes all filters, False otherwise.
    """

    def __init__(self, filters: Optional[list[TrackFilter]] = None):
        self.filters = filters or []

    def __call__(self, track: Track) -> bool:
        """Apply all filters in sequence."""
        return all(f(track) for f in self.filters)
