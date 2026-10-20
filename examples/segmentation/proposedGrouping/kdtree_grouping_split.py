import math
import numpy as np
from tqdm import tqdm


class KDTreeNode:
    def __init__(self, points, indices, depth=0, bounds_min=None, bounds_max=None):
        self.points = points
        self.indices = indices
        self.left = None
        self.right = None
        self.axis = depth % 3
        self.depth = depth
        self.bounds_min = (
            np.asarray(bounds_min, dtype=float).copy()
            if bounds_min is not None else points.min(axis=0).astype(float)
        )
        self.bounds_max = (
            np.asarray(bounds_max, dtype=float).copy()
            if bounds_max is not None else points.max(axis=0).astype(float)
        )

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_spans(self):
        """Return X, Y, Z spans of this node's spatial bounding box."""
        return self.bounds_max - self.bounds_min

    def get_spatial_size(self):
        """Return bounding-box surface area (not 3D volume)."""
        x, y, z = self.get_spans()
        return float(2 * (x * y + x * z + y * z))


def split_node(node, axis):
    """Split a node at the median along axis; update child spatial bounds."""
    points, indices = node.points, node.indices
    if len(points) <= 1:
        return False

    sorted_idx = np.argsort(points[:, axis])
    sorted_points = points[sorted_idx]
    sorted_indices = indices[sorted_idx]
    mid = len(sorted_points) // 2

    left_points, right_points = sorted_points[:mid], sorted_points[mid:]
    left_indices, right_indices = sorted_indices[:mid], sorted_indices[mid:]
    if len(left_points) == 0 or len(right_points) == 0:
        return False

    split_value = (float(left_points[-1, axis]) + float(right_points[0, axis])) / 2.0
    left_min, left_max = node.bounds_min.copy(), node.bounds_max.copy()
    right_min, right_max = node.bounds_min.copy(), node.bounds_max.copy()
    left_max[axis] = split_value
    right_min[axis] = split_value

    node.left = KDTreeNode(left_points, left_indices, node.depth + 1, left_min, left_max)
    node.right = KDTreeNode(right_points, right_indices, node.depth + 1, right_min, right_max)
    node.axis = axis
    return True


def build_kdtree(points, threshold, depth=0, pbar=None):
    points = np.asarray(points)
    if len(points) == 0:
        raise ValueError("Cannot build a KD-tree with zero points.")
    indices = np.arange(points.shape[0])
    bounds_min = points.min(axis=0).astype(float)
    bounds_max = points.max(axis=0).astype(float)
    return _build_kdtree(points, indices, threshold, depth, pbar, bounds_min, bounds_max)


def _build_kdtree(points, indices, threshold, depth=0, pbar=None,
                  bounds_min=None, bounds_max=None):
    if pbar is not None:
        pbar.update(1)
    node = KDTreeNode(points, indices, depth, bounds_min, bounds_max)
    if points.shape[0] <= threshold:
        return node

    # Preserve alternating X/Y/Z splitting used by the supplied implementation.
    axis = depth % 3
    node.axis = axis
    sorted_idx = np.argsort(points[:, axis])
    sorted_points, sorted_indices = points[sorted_idx], indices[sorted_idx]
    mid = len(sorted_points) // 2
    left_points, right_points = sorted_points[:mid], sorted_points[mid:]
    left_indices, right_indices = sorted_indices[:mid], sorted_indices[mid:]
    if len(left_points) == 0 or len(right_points) == 0:
        return node

    split_value = (float(left_points[-1, axis]) + float(right_points[0, axis])) / 2.0
    left_min, left_max = node.bounds_min.copy(), node.bounds_max.copy()
    right_min, right_max = node.bounds_min.copy(), node.bounds_max.copy()
    left_max[axis] = split_value
    right_min[axis] = split_value

    node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1,
                              pbar, left_min, left_max)
    node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1,
                               pbar, right_min, right_max)
    return node


def get_leaf_nodes(node):
    leaves = []
    def _collect(n):
        if n.is_leaf():
            leaves.append(n)
        else:
            if n.left is not None:
                _collect(n.left)
            if n.right is not None:
                _collect(n.right)
    _collect(node)
    return leaves


def get_leaf_indices(node):
    return [leaf.indices for leaf in get_leaf_nodes(node)]


def refine_largest_leaves(root, alpha=0):
    """Split once the alpha percent largest splittable leaves, by surface area."""
    if not 0 <= alpha <= 100:
        raise ValueError("alpha must be between 0 and 100.")
    leaves = get_leaf_nodes(root)
    splittable = [leaf for leaf in leaves if len(leaf.points) >= 2]
    if alpha == 0 or not splittable:
        return []

    ranked = sorted(splittable, key=lambda leaf: leaf.get_spatial_size(), reverse=True)
    num_to_split = min(math.ceil(len(leaves) * alpha / 100.0), len(ranked))
    refined = []
    for leaf in ranked[:num_to_split]:
        split_axis = int(np.argmax(leaf.get_spans()))
        if split_node(leaf, split_axis):
            refined.append(leaf)
    return refined


def round_robin(numpy_list):
    if not numpy_list:
        return []
    max_len = max(len(group) for group in numpy_list)
    chunks = []
    for i in range(max_len):
        chunk = [group[i % len(group)] for group in numpy_list]
        chunks.append(chunk)
    return chunks


def proposed_grouping(coord, threshold, alpha=0, return_bucket_sizes=False):
    points = np.asarray(coord)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coord must have shape (N, 3).")
    if threshold < 1:
        raise ValueError("threshold must be at least 1.")

    with tqdm(desc="CPU KD-Tree", unit="node") as pbar:
        root = build_kdtree(points, threshold, pbar=pbar)

    refined = refine_largest_leaves(root, alpha=alpha)
    print(f"Alpha: {alpha}%")
    print(f"Leaves refined: {len(refined)}")
    leaf_nodes = get_leaf_nodes(root)
    print(f"Final number of leaves: {len(leaf_nodes)}")

    if leaf_nodes:
        sizes = np.asarray([leaf.get_spatial_size() for leaf in leaf_nodes])
        print(f"Smallest leaf bucket surface area: {sizes.min():.6f}")
        print(f"Largest leaf bucket surface area:  {sizes.max():.6f}")
        print(f"Mean leaf bucket surface area:     {sizes.mean():.6f}")

    leaf_indices = [leaf.indices for leaf in leaf_nodes]
    if not return_bucket_sizes:
        return leaf_indices

    # One surface-area value per input point, in the original point order.
    point_bucket_sizes = np.empty(points.shape[0], dtype=np.float64)
    assigned = np.zeros(points.shape[0], dtype=bool)
    for leaf in leaf_nodes:
        area = leaf.get_spatial_size()
        point_bucket_sizes[leaf.indices] = area
        assigned[leaf.indices] = True
    if not np.all(assigned):
        missing = np.flatnonzero(~assigned)
        raise RuntimeError(f"KD-tree did not assign bucket sizes to {len(missing)} points.")
    return leaf_indices, point_bucket_sizes


def create_chunks(coord, threshold, alpha=0, return_bucket_sizes=False):
    print("Create chunks")
    grouping_result = proposed_grouping(
        coord, threshold, alpha=alpha, return_bucket_sizes=return_bucket_sizes
    )
    if return_bucket_sizes:
        numpy_list, point_bucket_sizes = grouping_result
    else:
        numpy_list = grouping_result

    chunks = [np.asarray(chunk) for chunk in round_robin(numpy_list)]
    if return_bucket_sizes:
        return chunks, numpy_list, point_bucket_sizes
    return chunks, numpy_list