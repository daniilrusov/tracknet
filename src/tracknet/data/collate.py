import torch
from .schemas import Track, BatchSample


def collate_fn_trackml(batch: list[Track]) -> BatchSample:
    """
    Collate function for batching Track objects for TrackNET training,
    which predicts the next hit location given the previous hits.

    Args:
        batch (List[Track]): A list of Track objects to be batched.

    Returns:
        BatchSample: A dictionary containing:
            - inputs (torch.Tensor): Zero-padded tensor of shape (batch_size, max_seq_len, 3) 
              containing the input sequences.
            - input_lengths (List[int]): A list of integers indicating the number of valid steps 
              in each sequence.
            - targets (torch.Tensor): Zero-padded tensor of shape (batch_size, max_seq_len, 3) 
              containing the target sequences.
            - target_mask (torch.Tensor): Boolean mask tensor of shape (batch_size, 2*max_seq_len-1)

    Steps:
    1) Identify the maximum (N-1) length among tracks in this batch.
    2) Create zero-padded tensors:
       - inputs:  shape (batch_size, max_seq_len, 3)
       - targets: shape (batch_size, max_seq_len, 3)
         where 3 = number of coordinates per hit (x, y, z).
    3) Populate these tensors with the shifted sequences from each track.
    4) Return BatchSample dictionary.
    """
    # 1) The max number of (input) steps = (max hits in track) - 1
    max_len = max(len(track.hits_xyz) for track in batch) - 1
    batch_size = len(batch)

    # 2) Prepare zero-padded tensors
    inputs = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    # Create mask tensor for valid target hits (non-padded positions)
    # *2 for t1 and t2 predictions
    # -1 because last hit has no t2 prediction
    target_mask = torch.zeros(batch_size, max_len * 2 - 1, dtype=torch.bool)
    input_lengths = []

    # 3) Fill the tensors
    for i, track in enumerate(batch):
        hits = track.hits_xyz  # shape (N, 3)
        n = len(hits) - 1      # number of training steps for this track
        # input: hits[:-1], target: hits[1:]
        inputs[i, :n, :] = torch.tensor(hits[:-1], dtype=torch.float32)
        targets[i, :n, :] = torch.tensor(hits[1:],  dtype=torch.float32)
        # Set mask for valid positions (both t1 and t2 predictions)
        target_mask[i, :n] = True  # t1 predictions
        # t2 predictions (n-1 because last hit has no t2)
        target_mask[i, max_len:max_len+n-1] = True
        input_lengths.append(n)

    # 4) Return batched data
    return BatchSample(
        inputs=inputs,
        input_lengths=input_lengths,
        targets=targets,
        target_mask=target_mask
    )



def collate_fn(batch: list[Track]) -> BatchSample:
    """
    Collate function for straw-tube TrackNET training.

    The model receives parameters of tubes already crossed by the particle and
    predicts the concrete next tube class.

    Args:
        batch (List[Track]): A list of Track objects to be batched.

    Returns:
        BatchSample: A dictionary containing:
            - inputs (torch.Tensor): Zero-padded tensor of shape (batch_size, max_seq_len, 3) 
              containing the input sequences.
            - input_lengths (List[int]): A list of integers indicating the number of valid steps 
              in each sequence.
            - targets (torch.Tensor): Zero-padded tensor of shape (batch_size, max_seq_len)
              containing next tube class ids.
            - target_mask (torch.Tensor): Boolean mask tensor of shape (batch_size, max_seq_len)

    Steps:
    1) Identify the maximum (N-1) length among tracks in this batch.
    2) Create zero-padded tensors:
       - inputs:  shape (batch_size, max_seq_len, 3)
       - targets: shape (batch_size, max_seq_len)
    3) Populate these tensors with the shifted sequences from each track.
    4) Return BatchSample dictionary.
    """
    # 1) The max number of (input) steps = (max hits in track) - 1
    max_len = max(len(track.hits_xyz) for track in batch) - 1
    batch_size = len(batch)

    # 2) Prepare zero-padded tensors
    input_features = batch[0].hits_xyz.shape[1]
    inputs = torch.zeros(batch_size, max_len, input_features, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    input_lengths = []

    # 3) Fill the tensors
    for i, track in enumerate(batch):
        hits = track.hits_xyz  # shape (N, 6)
        tube_ids = track.tube_ids
        n = len(hits) - 1      # number of training steps for this track
        # input: hits[:-1], target: hits[1:]
        inputs[i, :n, :] = torch.tensor(hits[:-1], dtype=torch.float32)
        targets[i, :n] = torch.tensor(tube_ids[1:], dtype=torch.long)
        # Set mask for valid positions
        target_mask[i, :n] = True  # t1 predictions
        input_lengths.append(n)

    # 4) Return batched data
    return BatchSample(
        inputs=inputs,
        input_lengths=input_lengths,
        targets=targets,
        target_mask=target_mask
    )
