from pathlib import Path
from typing import Generator, Literal, Optional, Set
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .schemas import Track, StrawTrack
from .filters import TrackFilter, FilterPipeline


DRIFT_SIM_COLUMNS = [
    'ev_id', 'wireid', 'dr', 'lr', 'station', 'tr_id',
    'x', 'y', 'x0', 'y0', 'z0'
]


class BlacklistManager:
    """
    Manages blacklists for hits and particles per event.

    Attributes:
        blacklisted_hits (dict[str, Set[int]]): 
            A mapping from event_id -> set of blacklisted hit IDs.
        blacklisted_particles (dict[str, Set[int]]): 
            A mapping from event_id -> set of blacklisted particle IDs.

    Methods:
        __init__(blacklist_dir: Optional[Path]):
            Initializes the BlacklistManager and loads blacklists if a directory is provided.

        _load_blacklists(blacklist_dir: Path):
            Loads blacklisted hits and particles from CSV files in the specified directory.

        is_valid(event_id: str, hit_ids: np.ndarray, particle_id: int) -> bool:
            Checks if the given hit IDs and particle ID are valid (not blacklisted for that event).
    """

    def __init__(self, blacklist_dir: Optional[Path]):
        self.blacklisted_hits: dict[str, Set[int]] = {}
        self.blacklisted_particles: dict[str, Set[int]] = {}

        if blacklist_dir:
            self._load_blacklists(blacklist_dir)

    def _load_blacklists(self, blacklist_dir: Path):
        # Load blacklisted hits
        for file in blacklist_dir.glob("*-blacklist_hits.csv"):
            # Example filename: event000001000-blacklist_hits.csv
            event_id = file.stem.split(
                "-blacklist_hits")[0]  # e.g. "event000001000"
            if event_id not in self.blacklisted_hits:
                self.blacklisted_hits[event_id] = set()

            df = pd.read_csv(file)
            # Assume the file has a column named 'hit_id'
            self.blacklisted_hits[event_id].update(df['hit_id'].values)

        # Load blacklisted particles
        for file in blacklist_dir.glob("*-blacklist_particles.csv"):
            # Example filename: event000001000-blacklist_particles.csv
            event_id = file.stem.split(
                "-blacklist_particles")[0]  # e.g. "event000001000"
            if event_id not in self.blacklisted_particles:
                self.blacklisted_particles[event_id] = set()

            df = pd.read_csv(file)
            # Assume the file has a column named 'particle_id'
            self.blacklisted_particles[event_id].update(
                df['particle_id'].values)

    def is_valid(self, event_id: str, hit_ids: np.ndarray, particle_id: int) -> bool:
        """Check if the given hit IDs and particle ID are valid for the specified event."""
        # Fetch blacklisted sets for the event, or empty if not present
        black_hits_for_event = self.blacklisted_hits.get(event_id, set())
        black_parts_for_event = self.blacklisted_particles.get(event_id, set())

        # Return True if particle_id and hit_ids are not in blacklisted sets
        if particle_id in black_parts_for_event:
            return False
        if any(hit_id in black_hits_for_event for hit_id in hit_ids):
            return False
        return True


class TrackMLTracksDataset(IterableDataset):
    """
    Iterable dataset for TrackML tracks.

    Args:
        data_dirs (str | Path | list[str | Path]): Directory or list of directories containing the TrackML data files.
        blacklist_dir (Optional[str | Path]): Directory containing blacklist files. Default is None.
        transforms (Optional[list]): List of transformations to apply to each track. Default is None.
        filters (Optional[list[TrackFilter]]): List of TrackFilter instances to apply to each track. Default is None.
        validation_split (float): Fraction of data to use for validation. Default is 0.1.
        split (Literal["train", "validation"]): Dataset split to use ('train' or 'validation'). Default is 'train'.

    Examples:
        >>> dataset = TrackMLTracksDataset(
        ...     data_dirs=['/path/to/data1', '/path/to/data2'],
        ...     blacklist_dir='/path/to/blacklist',
        ...     transforms=[DropRepeatedLayerHits()],
        ...     filters=[MinHitsFilter(3), PtFilter(0.8)],
        ...     validation_split=0.3,
        ...     split='train'
        ... )
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        blacklist_dir: Optional[str | Path] = None,
        transforms: Optional[list] = None,
        filters: Optional[list[TrackFilter]] = None,
        validation_split: float = 0.1,
        split: Literal["train", "validation"] = 'train'
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.transforms = transforms or []
        self.filter_pipeline = FilterPipeline(filters)

        self.blacklist = BlacklistManager(
            Path(blacklist_dir) if blacklist_dir else None)

        all_files = []
        for data_dir in self.data_dirs:
            all_files.extend(sorted(data_dir.glob("event*-hits.csv")))

        n_val = int(len(all_files) * validation_split)

        if split == 'train':
            self.event_files = all_files[n_val:]
        elif split == 'validation':
            self.event_files = all_files[:n_val]
        else:
            raise ValueError(
                f"Invalid split value: {split}. Must be 'train' or 'validation'.")

    def _process_event(self, event_file: Path) -> list[Track]:
        from trackml.dataset import load_event

        event_id = event_file.stem.split('-')[0]
        data_dir = event_file.parent

        hits, particles, truth = load_event(
            data_dir / event_id,
            parts=['hits', 'particles', 'truth']
        )

        track_data = truth.merge(hits, on='hit_id').merge(
            particles, on='particle_id')
        tracks = []

        for particle_id, group in track_data.groupby('particle_id'):
            hit_ids = group['hit_id'].values

            if not self.blacklist.is_valid(event_id, hit_ids, particle_id):
                continue

            track = Track(
                # from trackml: The reconstructed tracks must be
                # uniquely identified only within each event.
                event_id=event_id,
                track_id=len(tracks),
                particle_id=particle_id,
                hits_xyz=group[['x', 'y', 'z']].values,
                px=group['px'].iloc[0],
                py=group['py'].iloc[0],
                pz=group['pz'].iloc[0],
                charge=group['q'].iloc[0],
                volume_ids=group['volume_id'].values,
                layer_ids=group['layer_id'].values,
                module_ids=group['module_id'].values,
                hit_ids=hit_ids
            )

            tracks.append(track)

        return tracks

    def __iter__(self) -> Generator[Track, None, None]:
        event_files = self.event_files
        worker_info = get_worker_info()
        if worker_info is not None:
            event_files = event_files[worker_info.id::worker_info.num_workers]

        for event_file in event_files:
            for track in self._process_event(event_file):
                for transform in self.transforms:
                    track = transform(track)
                if self.filter_pipeline(track):
                    yield track


class StrawTracksDataset(Dataset):
    """
    Map-style dataset for straw detector tracks.

    Args:
        data_dirs (str | Path | list[str | Path]): Directory or list of directories containing straw data files.
        blacklist_dir (Optional[str | Path]): Directory containing blacklist files. Default is None.
        transforms (Optional[list]): List of transformations to apply to each track. Default is None.
        filters (Optional[list[TrackFilter]]): List of TrackFilter instances to apply to each track. Default is None.
        validation_split (float): Fraction of data to use for validation. Default is 0.1.
        split (Literal["train", "validation"]): Dataset split to use ('train' or 'validation'). Default is 'train'.

    Examples:
        >>> dataset = StrawTracksDataset(
        ...     data_dirs=['/path/to/data1', '/path/to/data2'],
        ...     blacklist_dir='/path/to/blacklist',
        ...     filters=[MinHitsFilter(3), PtFilter(0.8)],
        ...     validation_split=0.3,
        ...     split='train'
        ... )
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        blacklist_dir: Optional[str | Path] = None,
        transforms: Optional[list] = None,
        filters: Optional[list[TrackFilter]] = None,
        validation_split: float = 0.1,
        split: Literal["train", "validation"] = 'train',
        data_format: Literal["drift_sim", "spd_prod4"] = "drift_sim",
        file_pattern: Optional[str] = None,
        input_columns: Optional[list[str]] = None,
        num_stations: int = 8,
        tubes_per_station: int = 151,
        tube_id_offset: int = 0,
        tube_id_mapping: Literal["station_modulo", "station_offset", "dense_wireid"] = "station_modulo",
        split_seed: int = 42,
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.transforms = transforms or []
        self.filter_pipeline = FilterPipeline(filters)
        self.data_format = data_format
        self.file_pattern = file_pattern or (
            "*.tsv" if data_format == "drift_sim"
            else "ana_r.MC2025_S1.minbias-P8-spdroot417-dev.10GeV-UU.PROD2025-004.RECO.1.*.root"
        )
        self.input_columns = input_columns or (
            ['x0', 'y0', 'z0', 'dr', 'lr', 'station']
            if data_format == "drift_sim"
            else ['wp1x', 'wp1y', 'wp1z', 'wp2x', 'wp2y', 'wp2z']
        )
        self.num_stations = num_stations
        self.tubes_per_station = tubes_per_station
        self.tube_id_offset = tube_id_offset
        self.tube_id_mapping = tube_id_mapping
        self.split = split
        self.split_seed = split_seed

        all_files = []
        for data_dir in self.data_dirs:
            all_files.extend(sorted(data_dir.glob(self.file_pattern)))

        all_dfs = []
        for file in all_files:
            all_dfs.append(self._read_straw_file(file))
        if len(all_dfs) == 0:
            raise FileNotFoundError(
                f"No straw data files matching '{self.file_pattern}' in {self.data_dirs}"
            )
        total_df = pd.concat(all_dfs, ignore_index=True)
        total_df['hit_id'] = list(range(len(total_df)))
        self.tube_id_to_class = self._build_tube_id_to_class(total_df)
        self.num_tubes = (
            len(self.tube_id_to_class)
            if self.tube_id_mapping == "dense_wireid"
            else self.num_stations * self.tubes_per_station
        )

        events_ids = total_df.ev_id.unique()
        split_rng = np.random.default_rng(self.split_seed)
        events_ids = split_rng.permutation(events_ids)

        n_val = int(len(events_ids) * validation_split)

        if split == 'train':
            self.events = total_df[total_df.ev_id.isin(events_ids[n_val:])]
            self.events_ids = events_ids[n_val:]
        elif split == 'validation':
            self.events = total_df[total_df.ev_id.isin(events_ids[:n_val])]
            self.events_ids = events_ids[:n_val]
        else:
            raise ValueError(
                f"Invalid split value: {split}. Must be 'train' or 'validation'.")
        self.events_by_id = {
            ev_id: event for ev_id, event in self.events.groupby('ev_id', sort=False)
        }
        self.tracks = self._build_tracks()

    def _read_straw_file(self, file: Path) -> pd.DataFrame:
        if self.data_format == "drift_sim":
            return pd.read_csv(file, sep=r"\s+", names=DRIFT_SIM_COLUMNS, engine="python")
        return pd.read_csv(file)

    def _build_tube_id_to_class(self, df: pd.DataFrame) -> dict[int, int]:
        if self.data_format == "drift_sim" and self.tube_id_mapping == "dense_wireid":
            tube_ids = sorted(df['wireid'].astype(int).unique())
            return {tube_id: class_id for class_id, tube_id in enumerate(tube_ids)}

        if self.data_format == "drift_sim":
            return {}

        if 'wireid' in df.columns:
            tube_ids = sorted(df['wireid'].astype(int).unique())
        else:
            tube_ids = sorted(df['hit_id'].astype(int).unique())
        return {tube_id: class_id for class_id, tube_id in enumerate(tube_ids)}

    def _tube_class_ids(self, group: pd.DataFrame) -> np.ndarray:
        if self.data_format == "drift_sim":
            if self.tube_id_mapping == "dense_wireid":
                wireids = group['wireid'].astype(int).to_numpy()
                class_ids = np.asarray([self.tube_id_to_class[int(wireid)] for wireid in wireids])
                return class_ids.astype(np.int64)

            station = group['station'].astype(int).to_numpy()
            if np.any(station < 1) or np.any(station > self.num_stations):
                bad_station = int(station[(station < 1) | (station > self.num_stations)][0])
                raise ValueError(
                    f"Station id {bad_station} is outside [1, {self.num_stations}]."
                )

            raw_tube_id = group['wireid'].astype(int).to_numpy() % 1000
            tube_id = raw_tube_id - self.tube_id_offset
            if self.tube_id_mapping == "station_modulo":
                tube_id = np.mod(tube_id, self.tubes_per_station)
            elif np.any((tube_id < 0) | (tube_id >= self.tubes_per_station)):
                bad_tube = int(tube_id[(tube_id < 0) | (tube_id >= self.tubes_per_station)][0])
                raise ValueError(
                    f"Tube id {bad_tube} is outside [0, {self.tubes_per_station}). "
                    "Use tube_id_mapping='station_modulo' or adjust tube_id_offset."
                )

            class_ids = (station - 1) * self.tubes_per_station + tube_id
            return class_ids.astype(np.int64)

        tube_column = 'wireid' if 'wireid' in group.columns else 'hit_id'
        tube_ids = group[tube_column].astype(int).to_numpy()
        return np.asarray([self.tube_id_to_class[int(tube_id)] for tube_id in tube_ids]).astype(np.int64)

    def _process_event(self, event_id, event) -> list[Track]:
        tracks = []

        for tr_id, group in event.groupby('tr_id'):
            if 'station' in group.columns:
                group = group.sort_values('station')
            elif 'z0' in group.columns:
                group = group.sort_values('z0')
            hit_ids = group['hit_id'].values
            tube_ids = self._tube_class_ids(group)

            track = StrawTrack(
                event_id=event_id,
                track_id=tr_id,
                hits_xyz=group[self.input_columns].values,
                tube_ids=tube_ids,
                hit_ids=hit_ids,
                hits_wp=(
                    group[['wpx', 'wpy', 'wpz']].values
                    if {'wpx', 'wpy', 'wpz'}.issubset(group.columns)
                    else None
                )
            )

            tracks.append(track)

        return tracks

    def _build_tracks(self) -> list[StrawTrack]:
        tracks = []
        for ev_id in self.events_ids:
            event = self.events_by_id[ev_id]
            for track in self._process_event(ev_id, event):
                for transform in self.transforms:
                    track = transform(track)
                if self.filter_pipeline(track):
                    tracks.append(track)
        return tracks

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> StrawTrack:
        return self.tracks[index]
