
import numpy as np
from tqdm import tqdm
import time
import math


class KDTreeNode:
    def __init__(self, points, indices, depth=0, bounds_min=None, bounds_max=None):
        self.points = points
        self.indices = indices

        self.left = None
        self.right = None

        self.axis = depth % 3
        self.depth = depth

        # Spatial bounds of this node
        self.bounds_min = (
            np.asarray(bounds_min, dtype=float).copy()
            if bounds_min is not None
            else points.min(axis=0).astype(float)
        )
        self.bounds_max = (
            np.asarray(bounds_max, dtype=float).copy()
            if bounds_max is not None
            else points.max(axis=0).astype(float)
        )

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_spans(self):
        """Return the X, Y, Z spatial spans."""
        return self.bounds_max - self.bounds_min

    # def get_spatial_size(self):
    #     """Longest side of this node's bounding box."""
    #     return float(np.max(self.get_spans()))
    def get_spatial_size(self):
        """Surface area of this node's bounding box."""
        x, y, z = self.get_spans()
        return float(2 * (x * y + x * z + y * z))


# ---------------------------------------------------
# Split helper
# ---------------------------------------------------
def split_node(node, axis):
    """
    Split a node at the median along the specified axis.
    The children inherit the parent's spatial bounds,
    except along the splitting dimension.
    """

    points = node.points
    indices = node.indices

    if len(points) <= 1:
        return False

    # Sort points along the selected dimension
    sorted_idx = np.argsort(points[:, axis])

    sorted_points = points[sorted_idx]
    sorted_indices = indices[sorted_idx]

    mid = len(sorted_points) // 2

    left_points = sorted_points[:mid]
    right_points = sorted_points[mid:]

    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid:]

    if len(left_points) == 0 or len(right_points) == 0:
        return False

    # Split plane between the two median-adjacent points
    split_value = (
        float(left_points[-1, axis]) +
        float(right_points[0, axis])
    ) / 2.0

    # Inherit all bounds from the parent
    left_min = node.bounds_min.copy()
    left_max = node.bounds_max.copy()

    right_min = node.bounds_min.copy()
    right_max = node.bounds_max.copy()

    # Only the splitting dimension changes
    left_max[axis] = split_value
    right_min[axis] = split_value

    node.left = KDTreeNode(
        left_points,
        left_indices,
        depth=node.depth + 1,
        bounds_min=left_min,
        bounds_max=left_max
    )

    node.right = KDTreeNode(
        right_points,
        right_indices,
        depth=node.depth + 1,
        bounds_min=right_min,
        bounds_max=right_max
    )

    node.axis = axis

    return True


# ---------------------------------------------------
# CPU KD-tree construction
# Original alternating-axis splitting is preserved
# ---------------------------------------------------
def build_kdtree(points, threshold, depth=0, pbar=None):
    points = np.asarray(points)

    indices = np.arange(points.shape[0])

    if len(points) == 0:
        raise ValueError("Cannot build a KD-tree with zero points.")

    bounds_min = points.min(axis=0).astype(float)
    bounds_max = points.max(axis=0).astype(float)

    return _build_kdtree(
        points,
        indices,
        threshold,
        depth,
        pbar,
        bounds_min,
        bounds_max
    )


def _build_kdtree(
    points,
    indices,
    threshold,
    depth=0,
    pbar=None,
    bounds_min=None,
    bounds_max=None
):
    if pbar is not None:
        pbar.update(1)

    node = KDTreeNode(
        points,
        indices,
        depth,
        bounds_min,
        bounds_max
    )

    if points.shape[0] <= threshold:
        return node

    # Preserve the original alternating-axis strategy
    axis = depth % 3

    node.axis = axis

    sorted_idx = np.argsort(points[:, axis])

    sorted_points = points[sorted_idx]
    sorted_indices = indices[sorted_idx]

    mid = len(sorted_points) // 2

    left_points = sorted_points[:mid]
    right_points = sorted_points[mid:]

    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid:]

    if len(left_points) == 0 or len(right_points) == 0:
        return node

    split_value = (
        float(left_points[-1, axis]) +
        float(right_points[0, axis])
    ) / 2.0

    # Copy the parent bounds to both children
    left_min = node.bounds_min.copy()
    left_max = node.bounds_max.copy()

    right_min = node.bounds_min.copy()
    right_max = node.bounds_max.copy()

    # Only the splitting dimension changes
    left_max[axis] = split_value
    right_min[axis] = split_value

    node.left = _build_kdtree(
        left_points,
        left_indices,
        threshold,
        depth + 1,
        pbar,
        left_min,
        left_max
    )

    node.right = _build_kdtree(
        right_points,
        right_indices,
        threshold,
        depth + 1,
        pbar,
        right_min,
        right_max
    )

    return node


# ---------------------------------------------------
# Collect leaf nodes
# ---------------------------------------------------
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


# ---------------------------------------------------
# Split the alpha percent largest leaves
# ---------------------------------------------------
def refine_largest_leaves(root, alpha=10):
    """
    Select the alpha percent largest splittable leaves,
    ranked by maximum spatial span, and split each once
    along its longest-span dimension.

    This performs one refinement pass; newly created
    children are not recursively refined.
    """

    if not 0 <= alpha <= 100:
        raise ValueError("alpha must be between 0 and 100.")

    leaves = get_leaf_nodes(root)

    # Only leaves containing at least two points can be split
    splittable_leaves = [
        leaf for leaf in leaves
        if len(leaf.points) >= 2
    ]

    if alpha == 0 or not splittable_leaves:
        return []

    # Largest spatial span first
    ranked_leaves = sorted(
        splittable_leaves,
        key=lambda leaf: leaf.get_spatial_size(),
        reverse=True
    )

    # Number of leaves to refine
    num_to_split = math.ceil(
        len(leaves) * alpha / 100.0
    )

    # Do not select more leaves than can be split
    num_to_split = min(num_to_split, len(ranked_leaves))

    selected_leaves = ranked_leaves[:num_to_split]

    refined = []

    for leaf in selected_leaves:
        spans = leaf.get_spans()

        # Split along the dimension with maximum spatial span
        split_axis = int(np.argmax(spans))

        if split_node(leaf, split_axis):
            refined.append(leaf)

    return refined


# ---------------------------------------------------
# Round-robin sampling (unchanged)
# ---------------------------------------------------
def round_robin(numpy_list):
    if not numpy_list:
        return []

    max_len = max(len(group) for group in numpy_list)

    chunks = []

    for i in range(max_len):
        chunk = []

        for group in numpy_list:
            chunk.append(group[i % len(group)])

        chunks.append(chunk)

    return chunks


# ---------------------------------------------------
# Proposed grouping
# ---------------------------------------------------
def proposed_grouping(coord, threshold, alpha=10):
    points = np.asarray(coord)

    rough_max_nodes = 10000

    with tqdm(total=rough_max_nodes, desc="CPU KD-Tree") as pbar:
        kdtree_root = build_kdtree(
            points,
            threshold,
            pbar=pbar
        )

    # Refine the largest alpha percent of leaves
    refined = refine_largest_leaves(
        kdtree_root,
        alpha=alpha
    )

    print(f"Alpha: {alpha}%")
    print(f"Leaves refined: {len(refined)}")

    # Collect the final leaves after refinement
    leaf_nodes = get_leaf_nodes(kdtree_root)

    print(f"Final number of leaves: {len(leaf_nodes)}")

    # Optional spatial-size summary
    if leaf_nodes:
        sizes = np.array([
            leaf.get_spatial_size()
            for leaf in leaf_nodes
        ])

        print(f"Smallest leaf span: {sizes.min():.6f}")
        print(f"Largest leaf span:  {sizes.max():.6f}")
        print(f"Mean leaf span:     {sizes.mean():.6f}")

    # Return the same kind of output as the original function
    return [leaf.indices for leaf in leaf_nodes]


# ---------------------------------------------------
# Create chunks
# ---------------------------------------------------
def create_chunks(coord, threshold, alpha=10):
    print("Create chunks")

    numpy_list = proposed_grouping(
        coord,
        threshold,
        alpha=alpha
    )

    chunks = round_robin(numpy_list)

    chunks = [np.array(chunk) for chunk in chunks]

    return chunks, numpy_list
