from typing import TypedDict
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
    """GRU encoder with a per-step classifier over straw tube ids."""

    def __init__(self,
                 input_features=6,
                 hidden_features=128,
                 num_tubes: int | None = None,
                 output_features: int | None = None,
                 batch_first=True):
        super().__init__()
        if num_tubes is None:
            num_tubes = output_features if output_features is not None else 8000

        self.input_features = input_features
        self.num_tubes = num_tubes
        self.rnn = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=2,
            batch_first=batch_first
        )

        self.tube_classifier = nn.Sequential(
            nn.Linear(hidden_features, num_tubes)
        )

    def forward(self, inputs: torch.Tensor, input_lengths: list[int]) -> StrawTubePrediction:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_features)
            input_lengths (list[int]): List of sequence lengths for each batch item

        Returns:
            StrawTubePrediction: A dictionary containing:
            - 'tube_logits_t1' (torch.Tensor): next-tube logits for every input step
        """
        x = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return StrawTubePrediction(
            tube_logits_t1=self.tube_classifier(x)
        )
