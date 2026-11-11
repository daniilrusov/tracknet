import pandas as pd
import torch

from .model import StrawTrackNET
from .straw_geometry import V3_TUBE_COUNTS


_MIN_CANDIDATE_HITS = 5


class StrawTrackNETInference:
    """Build track candidates from the hits of one DriftSim V3 event."""

    def __init__(self, model: StrawTrackNET, batch_size: int = 512):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.model = model.eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, event: pd.DataFrame) -> list[list[int]]:
        """Return candidate paths as positional row indices into ``event``.

        The event needs x0, y0, z0, dr, station and tube_class_id columns.
        Each path starts on stations 1 and 2. On the next station, each hit
        whose tube appears in the model's global top ten starts a branch.
        A path ends when it has no match. Only paths with at least five
        hits are returned, including the two seed hits in that count.
        """
        if event.empty:
            return []
        if "ev_id" in event and event["ev_id"].nunique() > 1:
            raise ValueError("Expected hits from a single event.")

        parameter = next(self.model.parameters())
        features = torch.as_tensor(
            event[["x0", "y0", "z0", "dr", "station"]].to_numpy(),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        tube_ids = torch.as_tensor(
            event["tube_class_id"].to_numpy(),
            dtype=torch.long,
            device=parameter.device,
        )
        stations = features[:, -1]
        station_hits = [
            (stations == station).nonzero(as_tuple=True)[0]
            for station in range(1, len(V3_TUBE_COUNTS) + 1)
        ]
        first, second = station_hits[:2]
        if first.numel() == 0 or second.numel() == 0:
            return []

        active = torch.cartesian_prod(first, second)
        completed: list[list[int]] = []
        for hits in station_hits[2:]:
            if active.numel() == 0 or hits.numel() == 0:
                break

            next_candidates = []
            hit_tubes = tube_ids[hits]
            candidate_length = active.size(1)
            for batch in active.split(self.batch_size):
                lengths = [candidate_length] * batch.size(0)
                logits = self.model(features[batch], lengths)["tube_logits_t1"][:, -1]
                top_tubes = logits.topk(min(10, logits.size(-1)), dim=-1).indices
                matches = (top_tubes[:, :, None] == hit_tubes).any(dim=1)

                if candidate_length >= _MIN_CANDIDATE_HITS:
                    completed.extend(batch[~matches.any(dim=1)].tolist())
                parents, matched_hits = matches.nonzero(as_tuple=True)
                next_candidates.append(
                    torch.cat((batch[parents], hits[matched_hits, None]), dim=1)
                )

            active = torch.cat(next_candidates, dim=0)

        if active.size(1) >= _MIN_CANDIDATE_HITS:
            completed.extend(active.tolist())
        return completed
