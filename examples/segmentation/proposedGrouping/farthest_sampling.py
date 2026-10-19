import torch
import numpy as np


@torch.no_grad()
def farthest_point_sampling(points, k, start_idx=0, device=None, return_numpy=False):
    """
    Greedy Farthest Point Sampling.

    Parameters
    ----------
    points : (N,D) numpy array or torch tensor
    k : number of points to sample (can be equal to N)
    start_idx : initial point index
    device : cpu/cuda
    return_numpy : return numpy arrays instead of tensors

    Returns
    -------
    sampled_points
    sampled_indices
    """

    input_is_numpy = isinstance(points, np.ndarray)

    if input_is_numpy:
        original_dtype = points.dtype

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        points_t = torch.from_numpy(points).to(device=device, dtype=torch.float32)

    elif isinstance(points, torch.Tensor):
        if device is None:
            device = points.device

        points_t = points.to(device=device, dtype=torch.float32)

    else:
        raise TypeError(f"Unsupported type {type(points)}")

    assert points_t.ndim == 2

    N, D = points_t.shape
    k = int(k)

    if k > N:
        raise ValueError(f"k ({k}) > N ({N})")

    sampled_indices = torch.empty((k,), dtype=torch.long, device=points_t.device)

    # Distance to the nearest selected point
    min_distances = torch.full((N,), float("inf"), device=points_t.device)

    # Prevent reselection
    selected = torch.zeros((N,), dtype=torch.bool, device=points_t.device)

    current = int(start_idx)

    for i in range(k):

        sampled_indices[i] = current
        selected[current] = True

        current_point = points_t[current]          # (D,)

        dist = ((points_t - current_point) ** 2).sum(dim=1)

        min_distances = torch.minimum(min_distances, dist)

        # Never choose an already selected point again
        min_distances[selected] = -1.0

        if i != k - 1:
            current = torch.argmax(min_distances).item()

    sampled_points = points_t[sampled_indices]

    if return_numpy:
        sampled_points = sampled_points.cpu().numpy()
        sampled_indices = sampled_indices.cpu().numpy()

        if input_is_numpy:
            sampled_points = sampled_points.astype(original_dtype, copy=False)

    return sampled_points, sampled_indices