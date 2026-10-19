import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points


@torch.no_grad()
def to_batched_torch(points, device="cuda", dtype=torch.float32):
    """
    Accepts numpy array or torch tensor of shape:
        (N, 3) or (B, N, 3)

    Returns torch tensor of shape:
        (B, N, 3)
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)

    points = points.to(device=device, dtype=dtype)

    if points.ndim == 2:
        points = points.unsqueeze(0)

    assert points.ndim == 3, "points should have shape (N, 3) or (B, N, 3)"
    assert points.shape[-1] == 3, "last dimension should be 3"

    return points


@torch.no_grad()
def internal_distance_stats(x):
    """
    x: Tensor of shape (K, 3)

    Returns internal pairwise distance statistics.
    """
    K = x.shape[0]

    # Pairwise distances: (K, K)
    d = torch.cdist(x, x, p=2)

    # Mask diagonal because distance from point to itself is zero
    diag_mask = torch.eye(K, dtype=torch.bool, device=x.device)

    pairwise_distances = d[~diag_mask]

    # Nearest neighbor distance for each point
    d_for_nn = d.masked_fill(diag_mask, float("inf"))
    nn_distances = d_for_nn.min(dim=1).values

    stats = {
        "min_pairwise": pairwise_distances.min().item(),
        "mean_pairwise": pairwise_distances.mean().item(),
        "median_pairwise": pairwise_distances.median().item(),
        "mean_nn": nn_distances.mean().item(),
        "median_nn": nn_distances.median().item(),
        "min_nn": nn_distances.min().item(),
        "max_nn": nn_distances.max().item(),
    }

    return stats


@torch.no_grad()
def cross_distance_stats(x, y):
    """
    x: Tensor of shape (K1, 3)
    y: Tensor of shape (K2, 3)

    Computes distances between two sets.
    """
    d = torch.cdist(x, y, p=2)

    x_to_y_nn = d.min(dim=1).values
    y_to_x_nn = d.min(dim=0).values

    stats = {
        "min_cross": d.min().item(),
        "mean_cross": d.mean().item(),
        "median_cross": d.median().item(),
        "mean_x_to_y_nn": x_to_y_nn.mean().item(),
        "mean_y_to_x_nn": y_to_x_nn.mean().item(),
        "median_x_to_y_nn": x_to_y_nn.median().item(),
        "median_y_to_x_nn": y_to_x_nn.median().item(),
    }

    return stats


def print_stats(title, stats):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in stats.items():
        print(f"{k:24s}: {v:.6f}")


@torch.no_grad()
def examine_fps_splitting(points, K=1000, random_start_point=False, device="cuda"):
    """
    Compares:

    Method A:
        FPS once with 2*K samples.
        Split into first K and second K.

    Method B:
        FPS with K samples.
        Remove those K points from the original cloud.
        FPS again with K samples from the remaining cloud.

    points:
        numpy array or torch tensor, shape (N, 3) or (1, N, 3)

    K:
        number of points in each subset

    random_start_point:
        PyTorch3D FPS random start flag.
        If False, the first K points from FPS(2K) should match FPS(K),
        assuming identical input order.

    Returns a dictionary with sampled points, indices, and stats.
    """

    points = to_batched_torch(points, device=device)

    assert points.shape[0] == 1, "This example assumes batch size B=1."
    B, N, C = points.shape

    assert 2 * K <= N, "Need at least 2*K points in the cloud."

    print(f"Input points: {points.shape}")
    print(f"K: {K}")
    print(f"Device: {points.device}")
    print(f"random_start_point: {random_start_point}")

    # ============================================================
    # Method A:
    # FPS once for 2*K, then split into first K and second K
    # ============================================================

    fps_2k_points, fps_2k_idx = sample_farthest_points(
        points,
        K=2 * K,
        random_start_point=random_start_point,
    )

    fps_2k_points = fps_2k_points[0]  # (2K, 3)
    fps_2k_idx = fps_2k_idx[0]        # (2K,)

    A1_points = fps_2k_points[:K]
    A2_points = fps_2k_points[K:2 * K]

    A1_idx = fps_2k_idx[:K]
    A2_idx = fps_2k_idx[K:2 * K]

    # ============================================================
    # Method B:
    # FPS for K, remove them, then FPS again for K
    # ============================================================

    fps_k_points, fps_k_idx = sample_farthest_points(
        points,
        K=K,
        random_start_point=random_start_point,
    )

    fps_k_points = fps_k_points[0]  # (K, 3)
    fps_k_idx = fps_k_idx[0]        # (K,)

    B1_points = fps_k_points
    B1_idx = fps_k_idx

    # Remove selected indices from original cloud
    mask = torch.ones(N, dtype=torch.bool, device=points.device)
    mask[B1_idx] = False

    remaining_points = points[0, mask, :]             # (N-K, 3)
    remaining_original_idx = torch.arange(N, device=points.device)[mask]

    remaining_points_batched = remaining_points.unsqueeze(0)

    fps_remaining_points, fps_remaining_local_idx = sample_farthest_points(
        remaining_points_batched,
        K=K,
        random_start_point=random_start_point,
    )

    B2_points = fps_remaining_points[0]              # (K, 3)
    B2_local_idx = fps_remaining_local_idx[0]        # indices inside remaining cloud
    B2_idx = remaining_original_idx[B2_local_idx]    # indices inside original cloud

    # ============================================================
    # Compare whether A1 and B1 are identical
    # ============================================================

    same_A1_B1_idx = torch.equal(A1_idx, B1_idx)
    same_A1_B1_points = torch.allclose(A1_points, B1_points)

    print("\nFirst K from FPS(2K) compared to FPS(K)")
    print("--------------------------------------")
    print("same indices:", same_A1_B1_idx)
    print("same points :", same_A1_B1_points)

    # ============================================================
    # Compute stats
    # ============================================================

    stats = {}

    stats["A1_internal"] = internal_distance_stats(A1_points)
    stats["A2_internal"] = internal_distance_stats(A2_points)
    stats["A1_A2_cross"] = cross_distance_stats(A1_points, A2_points)

    stats["B1_internal"] = internal_distance_stats(B1_points)
    stats["B2_internal"] = internal_distance_stats(B2_points)
    stats["B1_B2_cross"] = cross_distance_stats(B1_points, B2_points)

    # Print stats
    print_stats("Method A: first K from FPS(2K), internal distance", stats["A1_internal"])
    print_stats("Method A: second K from FPS(2K), internal distance", stats["A2_internal"])
    print_stats("Method A: cross distance between first K and second K", stats["A1_A2_cross"])

    print_stats("Method B: first FPS(K), internal distance", stats["B1_internal"])
    print_stats("Method B: second FPS(K) after removing first K, internal distance", stats["B2_internal"])
    print_stats("Method B: cross distance between first K and second K", stats["B1_B2_cross"])

    result = {
        "A1_points": A1_points,
        "A2_points": A2_points,
        "A1_idx": A1_idx,
        "A2_idx": A2_idx,

        "B1_points": B1_points,
        "B2_points": B2_points,
        "B1_idx": B1_idx,
        "B2_idx": B2_idx,

        "stats": stats,
    }

    return result

