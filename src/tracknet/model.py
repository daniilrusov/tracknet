from typing import TypedDict, NotRequired
import torch
import torch.nn as nn


class TrackPrediction(TypedDict):
    """
    TrackPrediction is a TypedDict that represents the prediction of tracking coordinates 
    and radii at two different time steps.

    Attributes:
        coords_t1 (torch.Tensor): Coordinates at time step t1 with shape 
            (batch_size, seq_len, output_features).
        radius_t1 (torch.Tensor): Radii at time step t1 with shape (batch_size, seq_len, 1).
        coords_t2 (torch.Tensor): Coordinates at time step t2 with shape 
            (batch_size, seq_len, output_features).
        radius_t2 (torch.Tensor): Radii at time step t2 with shape (batch_size, seq_len, 1).
    """
    coords_t1: torch.Tensor
    radius_t1: torch.Tensor
    coords_t2: torch.Tensor
    radius_t2: torch.Tensor


class PointPrediction(TypedDict):
    """
    PointPrediction is a TypedDict that represents the prediction of tracking coordinates.

    Attributes:
        coords_t1 (torch.Tensor): Coordinates at time step t1 with shape 
            (batch_size, seq_len, output_features).
    """
    coords_t1: torch.Tensor


class StrawTubePrediction(TypedDict):
    """
    Prediction for the next straw tube.

    Attributes:
        tube_logits_t1 (torch.Tensor): Class logits for the next tube with shape
            (batch_size, seq_len, num_tubes).
    """
    tube_logits_t1: torch.Tensor
    coords_t0: NotRequired[torch.Tensor]
    coords_t1: NotRequired[torch.Tensor]


class StepAheadTrackNET(nn.Module):
    """
    RNN that predicts two consecutive search areas for next hits.

    The model learns to extrapolate track trajectories by predicting spherical regions 
    at steps t+1 and t+2 based on the sequence of previous hits [0,t]. This dual prediction 
    allows handling missing detector hits: if no hit is found at t+1, the t+2 prediction 
    can be used to validate track candidates.

    Architecture:
        - GRU-based sequence encoder processes variable-length hit sequences
        - Parallel prediction heads for coordinates and search radii at t+1 and t+2
        - Coordinates are predicted directly, search radii use Softplus activation

    Args:
        input_features (int, optional): Number of input features per hit. 
            Defaults to 3 (x,y,z) coordinates.
        hidden_features (int, optional): Size of GRU hidden state. Defaults to 32.
        output_features (int, optional): Number of coordinate dimensions to predict. 
            Defaults to 3 (x,y,z) coordinates.
        batch_first (bool, optional): Input tensor format. Defaults to True.

    Outputs:
        TrackPrediction: A dictionary containing:
            - 'coords_t1' (torch.Tensor): Predicted coordinates for t+1
            - 'radius_t1' (torch.Tensor): Predicted search radius for t+1
            - 'coords_t2' (torch.Tensor): Predicted coordinates for t+2 
            - 'radius_t2' (torch.Tensor): Predicted search radius for t+2
    """

    def __init__(self,
                 input_features=3,
                 hidden_features=32,
                 output_features=3,
                 batch_first=True):
        super().__init__()
        self.input_features = input_features
        self.rnn = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=2,
            batch_first=batch_first
        )

        # outputs for two hits simultaneously:
        # for t+1 and t+2 based on [0, t] input hits
        self.coords_1 = nn.Sequential(
            nn.Linear(hidden_features, output_features)
        )
        self.radius_1 = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.Softplus()
        )
        self.coords_2 = nn.Sequential(
            nn.Linear(hidden_features, output_features)
        )
        self.radius_2 = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.Softplus()
        )

    def forward(self, inputs: torch.Tensor, input_lengths: list[int]) -> TrackPrediction:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_features)
            input_lengths (list[int]): List of sequence lengths for each batch item

        Returns:
            TrackPrediction: A dictionary containing:
            - 'coords_t1' (torch.Tensor): Predicted coordinates for t+1
            - 'radius_t1' (torch.Tensor): Predicted search radius for t+1
            - 'coords_t2' (torch.Tensor): Predicted coordinates for t+2
            - 'radius_t2' (torch.Tensor): Predicted search radius for t+2
        """
        x = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return TrackPrediction(
            coords_t1=self.coords_1(x),
            radius_t1=self.radius_1(x),
            coords_t2=self.coords_2(x),
            radius_t2=self.radius_2(x)
        )


class StrawTrackNET(nn.Module):
    """Causal GRU with a next-tube classifier and optional particle xyz heads.

    The current straw input schema is ``[x0, y0, z0, dr, station]``. Continuous
    detector coordinates are normalized to roughly ``[-1, 1]`` and the station
    number is used only to select a learned X/Y/U/V plane embedding. The legacy
    six-feature path is retained for loading checkpoints created before this
    representation was introduced. Coordinate outputs are in detector units:
    ``coords_t0`` describes the last input hit and ``coords_t1`` the next hit
    whose tube is classified by ``tube_logits_t1`` at the same step.
    """

    def __init__(self,
                 input_features=5,
                 hidden_features=128,
                 num_tubes: int | None = None,
                 output_features: int | None = None,
                 batch_first=True,
                 use_plane_embedding: bool = True,
                 plane_embedding_dim: int = 8,
                 continuous_feature_center=(0.0, 0.0, 120.0, 2.5),
                 continuous_feature_scale=(750.0, 750.0, 120.0, 2.5),
                 predict_coordinates: bool = False,
                 coordinate_center=(0.0, 0.0, 120.0),
                 coordinate_scale=(750.0, 750.0, 120.0)):
        super().__init__()
        if num_tubes is None:
            num_tubes = output_features if output_features is not None else 8000
        if use_plane_embedding and input_features != 5:
            raise ValueError(
                "Plane-embedded straw inputs must contain exactly "
                "[x0, y0, z0, dr, station]."
            )
        if plane_embedding_dim <= 0:
            raise ValueError("plane_embedding_dim must be positive.")

        self.input_features = input_features
        self.num_tubes = num_tubes
        self.use_plane_embedding = bool(use_plane_embedding)
        self.plane_embedding_dim = int(plane_embedding_dim)
        self.predict_coordinates = bool(predict_coordinates)
        if self.predict_coordinates:
            center = torch.as_tensor(coordinate_center, dtype=torch.float32)
            scale = torch.as_tensor(coordinate_scale, dtype=torch.float32)
            if center.shape != (3,) or scale.shape != (3,):
                raise ValueError("Coordinate center and scale must contain x, y, z values.")
            if (not torch.isfinite(center).all()
                    or not torch.isfinite(scale).all() or torch.any(scale <= 0)):
                raise ValueError("Coordinate center must be finite and scales finite and positive.")
            self.register_buffer("coordinate_center", center)
            self.register_buffer("coordinate_scale", scale)
            self.coords_current = nn.Linear(hidden_features, 3)
            self.coords_next = nn.Linear(hidden_features, 3)

        rnn_input_features = input_features
        if self.use_plane_embedding:
            center = torch.as_tensor(continuous_feature_center, dtype=torch.float32)
            scale = torch.as_tensor(continuous_feature_scale, dtype=torch.float32)
            if center.shape != (4,) or scale.shape != (4,):
                raise ValueError(
                    "continuous_feature_center and continuous_feature_scale "
                    "must each contain four values for x0, y0, z0 and dr."
                )
            if torch.any(scale <= 0):
                raise ValueError("All continuous feature scales must be positive.")
            self.register_buffer("continuous_feature_center", center)
            self.register_buffer("continuous_feature_scale", scale)
            self.plane_embedding = nn.Embedding(4, self.plane_embedding_dim)
            rnn_input_features = 4 + self.plane_embedding_dim

        self.rnn_input_features = rnn_input_features
        self.rnn = nn.GRU(
            input_size=rnn_input_features,
            hidden_size=hidden_features,
            num_layers=2,
            batch_first=batch_first
        )

        self.tube_classifier = nn.Sequential(
            nn.Linear(hidden_features, num_tubes)
        )

    def encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize continuous inputs and replace station with an XYUV embedding."""
        if inputs.size(-1) != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features, got {inputs.size(-1)}."
            )
        if not self.use_plane_embedding:
            return inputs

        continuous = (
            inputs[..., :4] - self.continuous_feature_center
        ) / self.continuous_feature_scale
        station = inputs[..., 4]
        rounded_station = station.round()
        non_padding = station != 0
        invalid_station = non_padding & (
            (station - rounded_station).abs() > 1e-4
        )
        invalid_station |= non_padding & (
            (rounded_station < 1) | (rounded_station > 8)
        )
        if torch.any(invalid_station):
            bad_station = float(station[invalid_station][0].detach().cpu())
            raise ValueError(f"Station id {bad_station} is outside integer range [1, 8].")

        # Stations 1/5, 2/6, 3/7 and 4/8 share X, Y, U and V embeddings.
        plane_ids = (rounded_station.long() - 1).remainder(4)
        plane_features = self.plane_embedding(plane_ids)
        return torch.cat((continuous, plane_features), dim=-1)

    def forward(self, inputs: torch.Tensor, input_lengths: list[int]) -> StrawTubePrediction:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_features)
            input_lengths (list[int]): List of sequence lengths for each batch item

        Returns:
            StrawTubePrediction: A dictionary containing:
            - 'tube_logits_t1' (torch.Tensor): next-tube logits for every input step
            - 'coords_t0', 'coords_t1': optional current/next particle xyz
        """
        x = self.encode_inputs(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return self.predict_from_hidden(x)

    def predict_from_hidden(self, hidden: torch.Tensor) -> StrawTubePrediction:
        """Predict tube t+1 and particle positions at t and t+1 in detector units.

        Both coordinate heads use only the causal prefix ending at t. Truth
        coordinates and the future tube's geometry are never model inputs.
        """
        output = StrawTubePrediction(tube_logits_t1=self.tube_classifier(hidden))
        if self.predict_coordinates:
            output['coords_t0'] = self.coords_current(hidden) * self.coordinate_scale + self.coordinate_center
            output['coords_t1'] = self.coords_next(hidden) * self.coordinate_scale + self.coordinate_center
        return output
