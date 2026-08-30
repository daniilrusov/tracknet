import hashlib
from pathlib import Path
from typing import Generator, Literal, Optional, Set
import numpy as np
import pandas as pd
import torch
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


class StreamingStrawTracksDataset(IterableDataset):
    """
    Streaming dataset for large drift-sim straw files.

    Unlike StrawTracksDataset, this class does not concatenate all input files or
    build the full track list in memory. It reads TSV/CSV files in chunks, keeps
    only one incomplete event between chunks, and yields tracks lazily.
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        blacklist_dir: Optional[str | Path] = None,
        transforms: Optional[list] = None,
        filters: Optional[list[TrackFilter]] = None,
        validation_split: float = 0.1,
        split: Literal["train", "validation"] = "train",
        data_format: Literal["drift_sim", "spd_prod4"] = "drift_sim",
        file_pattern: Optional[str] = None,
        input_columns: Optional[list[str]] = None,
        num_stations: int = 8,
        tubes_per_station: int = 151,
        tube_id_offset: int = 0,
        tube_id_mapping: Literal["station_modulo", "station_offset"] = "station_modulo",
        split_seed: int = 42,
        chunk_size: int = 1_000_000,
    ):
        if blacklist_dir is not None:
            raise ValueError("StreamingStrawTracksDataset does not support blacklist_dir.")
        if tube_id_mapping == "dense_wireid":
            raise ValueError("StreamingStrawTracksDataset does not support dense_wireid mapping.")
        if split not in ("train", "validation"):
            raise ValueError(
                f"Invalid split value: {split}. Must be 'train' or 'validation'."
            )

        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.transforms = transforms or []
        self.filter_pipeline = FilterPipeline(filters)
        self.validation_split = validation_split
        self.split = split
        self.data_format = data_format
        self.file_pattern = file_pattern or (
            "*.tsv" if data_format == "drift_sim"
            else "ana_r.MC2025_S1.minbias-P8-spdroot417-dev.10GeV-UU.PROD2025-004.RECO.1.*.root"
        )
        self.input_columns = input_columns or (
            ["x0", "y0", "z0", "dr", "lr", "station"]
            if data_format == "drift_sim"
            else ["wp1x", "wp1y", "wp1z", "wp2x", "wp2y", "wp2z"]
        )
        self.num_stations = num_stations
        self.tubes_per_station = tubes_per_station
        self.tube_id_offset = tube_id_offset
        self.tube_id_mapping = tube_id_mapping
        self.split_seed = split_seed
        self.chunk_size = chunk_size
        self.num_tubes = self.num_stations * self.tubes_per_station

        self.files = []
        for data_dir in self.data_dirs:
            self.files.extend(sorted(data_dir.glob(self.file_pattern)))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No straw data files matching '{self.file_pattern}' in {self.data_dirs}"
            )

    def _event_hash_int(self, event_id, salt: str) -> int:
        payload = f"{salt}:{self.split_seed}:{event_id}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    def _event_matches_split(self, event_id) -> bool:
        value = self._event_hash_int(event_id, "split") / 2**64
        is_validation = value < self.validation_split
        return is_validation if self.split == "validation" else not is_validation

    def _event_matches_worker(self, event_id, worker_id: int, num_workers: int) -> bool:
        if num_workers <= 1:
            return True
        return self._event_hash_int(event_id, "worker") % num_workers == worker_id

    def _read_file_chunks(self, file: Path):
        if self.data_format == "drift_sim":
            return pd.read_csv(
                file,
                sep=r"\s+",
                names=DRIFT_SIM_COLUMNS,
                engine="python",
                chunksize=self.chunk_size,
            )
        return pd.read_csv(file, chunksize=self.chunk_size)

    def _tube_class_ids(self, group: pd.DataFrame) -> np.ndarray:
        if self.data_format == "drift_sim":
            station = group["station"].astype(int).to_numpy()
            if np.any(station < 1) or np.any(station > self.num_stations):
                bad_station = int(station[(station < 1) | (station > self.num_stations)][0])
                raise ValueError(
                    f"Station id {bad_station} is outside [1, {self.num_stations}]."
                )

            raw_tube_id = group["wireid"].astype(int).to_numpy() % 1000
            tube_id = raw_tube_id - self.tube_id_offset
            if self.tube_id_mapping == "station_modulo":
                tube_id = np.mod(tube_id, self.tubes_per_station)
            elif np.any((tube_id < 0) | (tube_id >= self.tubes_per_station)):
                bad_tube = int(tube_id[(tube_id < 0) | (tube_id >= self.tubes_per_station)][0])
                raise ValueError(
                    f"Tube id {bad_tube} is outside [0, {self.tubes_per_station}). "
                    "Use tube_id_mapping='station_modulo' or adjust tube_id_offset."
                )

            return ((station - 1) * self.tubes_per_station + tube_id).astype(np.int64)

        tube_column = "wireid" if "wireid" in group.columns else "hit_id"
        return group[tube_column].astype(int).to_numpy().astype(np.int64)

    def _process_event(self, event_id, event) -> list[StrawTrack]:
        tracks = []
        for tr_id, group in event.groupby("tr_id", sort=False):
            if "station" in group.columns:
                group = group.sort_values("station")
            elif "z0" in group.columns:
                group = group.sort_values("z0")

            track = StrawTrack(
                event_id=event_id,
                track_id=tr_id,
                hits_xyz=group[self.input_columns].values,
                tube_ids=self._tube_class_ids(group),
                hit_ids=group["hit_id"].values,
                hits_wp=(
                    group[["wpx", "wpy", "wpz"]].values
                    if {"wpx", "wpy", "wpz"}.issubset(group.columns)
                    else None
                ),
            )
            tracks.append(track)
        return tracks

    def _iter_complete_events(self, df: pd.DataFrame):
        for event_id, event in df.groupby("ev_id", sort=False):
            yield event_id, event

    def _iter_file_tracks(self, file: Path, worker_id: int, num_workers: int):
        carry = None
        next_hit_id = 0

        for chunk in self._read_file_chunks(file):
            chunk = chunk.copy()
            chunk["hit_id"] = np.arange(next_hit_id, next_hit_id + len(chunk))
            next_hit_id += len(chunk)

            if carry is not None and len(carry) > 0:
                chunk = pd.concat([carry, chunk], ignore_index=True)

            last_event_id = chunk["ev_id"].iloc[-1]
            complete = chunk[chunk["ev_id"] != last_event_id]
            carry = chunk[chunk["ev_id"] == last_event_id]

            if len(complete) == 0:
                continue

            for event_id, event in self._iter_complete_events(complete):
                if not self._event_matches_split(event_id):
                    continue
                if not self._event_matches_worker(event_id, worker_id, num_workers):
                    continue
                for track in self._process_event(event_id, event):
                    for transform in self.transforms:
                        track = transform(track)
                    if self.filter_pipeline(track):
                        yield track

        if carry is not None and len(carry) > 0:
            for event_id, event in self._iter_complete_events(carry):
                if not self._event_matches_split(event_id):
                    continue
                if not self._event_matches_worker(event_id, worker_id, num_workers):
                    continue
                for track in self._process_event(event_id, event):
                    for transform in self.transforms:
                        track = transform(track)
                    if self.filter_pipeline(track):
                        yield track

    def __iter__(self) -> Generator[StrawTrack, None, None]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        for file in self.files:
            for track in self._iter_file_tracks(file, worker_id, num_workers):
                yield track


class PrebatchedStrawTracksDataset(IterableDataset):
    """
    Fast iterable dataset for preprocessed drift-sim shards.

    Expected shard format is produced by scripts/preprocess_drift_sim.py. Each
    shard contains fixed-size tensors for inputs, targets, masks, and lengths,
    so training does not spend every step parsing TSV files with pandas.
    """

    prebatched = True

    def __init__(
        self,
        cache_dir: str | Path,
        batch_size: int,
        split: Literal["train", "validation"] = "train",
        num_tubes: int = 8 * 151,
        shuffle_shards: bool = False,
        shuffle_tracks: bool = False,
        split_seed: int = 42,
        schema_version: str | int | None = None,
        **_,
    ):
        if split not in ("train", "validation"):
            raise ValueError(
                f"Invalid split value: {split}. Must be 'train' or 'validation'."
            )

        self.cache_dir = Path(cache_dir)
        self.split = split
        self.batch_size = int(batch_size)
        self.num_tubes = int(num_tubes)
        self.shuffle_shards = bool(shuffle_shards) and split == "train"
        self.shuffle_tracks = bool(shuffle_tracks) and split == "train"
        self.split_seed = int(split_seed)
        self._iteration = 0

        metadata_path = self.cache_dir / "metadata.pt"
        if schema_version is not None and not metadata_path.exists():
            raise FileNotFoundError(
                f"Dataset schema {schema_version} requires cache metadata: "
                f"{metadata_path}"
            )
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location="cpu")
            metadata_num_tubes = metadata.get("num_tubes")
            if metadata_num_tubes is not None and int(metadata_num_tubes) != self.num_tubes:
                raise ValueError(
                    f"Dataset config has num_tubes={self.num_tubes}, but cache "
                    f"metadata has num_tubes={metadata_num_tubes}."
                )
            if schema_version is not None:
                expected_schema = (
                    "v3"
                    if str(schema_version).lower() in ("3", "v3")
                    else str(schema_version)
                )
                actual_schema = str(metadata.get("schema_version", "legacy")).lower()
                if actual_schema != expected_schema:
                    raise ValueError(
                        f"Dataset config expects schema {expected_schema}, but cache "
                        f"metadata reports {actual_schema}."
                    )

        split_dir = self.cache_dir / self.split
        self.shard_files = sorted(split_dir.glob("*.pt"))
        if len(self.shard_files) == 0:
            raise FileNotFoundError(
                f"No preprocessed {self.split} shards found in {split_dir}. "
                "Run: python scripts/preprocess_drift_sim.py "
                "--input-dir outputs/drift_sim --output-dir outputs/drift_sim_cache"
            )

    def _worker_shards(self, iteration: int) -> list[Path]:
        shard_files = self.shard_files
        if self.shuffle_shards:
            rng = np.random.default_rng(self.split_seed + iteration)
            shard_files = list(rng.permutation(shard_files))

        worker_info = get_worker_info()
        if worker_info is None:
            return list(shard_files)
        return list(shard_files[worker_info.id::worker_info.num_workers])

    def __iter__(self):
        iteration = self._iteration
        self._iteration += 1
        for shard_file in self._worker_shards(iteration):
            shard = torch.load(shard_file, map_location="cpu")
            inputs = shard["inputs"].float()
            targets = shard["targets"].long()
            target_mask = shard["target_mask"].bool()
            input_lengths = shard["input_lengths"].long()
            n_tracks = inputs.size(0)
            order = None
            if self.shuffle_tracks:
                payload = (
                    f"track-order:{self.split_seed}:{iteration}:{shard_file.name}"
                ).encode("utf-8")
                order_seed = int.from_bytes(
                    hashlib.blake2b(payload, digest_size=8).digest(), "big"
                )
                generator = torch.Generator().manual_seed(order_seed)
                order = torch.randperm(n_tracks, generator=generator)

            for start in range(0, n_tracks, self.batch_size):
                end = min(start + self.batch_size, n_tracks)
                selection = slice(start, end) if order is None else order[start:end]
                yield {
                    "inputs": inputs[selection],
                    "targets": targets[selection],
                    "target_mask": target_mask[selection],
                    "input_lengths": input_lengths[selection].tolist(),
                }
