import numpy as np
from tqdm import tqdm
import time


class KDTreeNode:
    def __init__(self, points, indices, depth=0):
        self.points = points
        self.indices = indices
        self.left = None
        self.right = None
        self.axis = None
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None


# ---------------------------------------------------
# CPU KD-tree
# Hybrid axis selection:
#   - Use max-variance axis if it is >30% larger
#   - Otherwise use cyclic x->y->z based on parent
# ---------------------------------------------------
def build_kdtree(points, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)

    indices = np.arange(points.shape[0])
    return _build_kdtree(points, indices, threshold, depth, parent_axis=2, pbar=pbar)
    # parent_axis=2 makes the root start with axis 0 (x)


def _build_kdtree(points,
                  indices,
                  threshold,
                  depth=0,
                  parent_axis=2,
                  pbar=None):

    node = KDTreeNode(points, indices, depth)

    if points.shape[0] <= threshold:
        return node

    # -------------------------------------------------
    # Compute variance along x,y,z
    # -------------------------------------------------
    variances = np.var(points, axis=0)

    max_axis = np.argmax(variances)

    # -------------------------------------------------
    # Decide whether variances are sufficiently different
    # -------------------------------------------------
    max_var = variances[max_axis]
    min_var = np.min(variances)

    if min_var == 0:
        ratio = np.inf if max_var > 0 else 1.0
    else:
        ratio = max_var / min_var

    # If largest variance is >30% larger than smallest,
    # use variance-based split.
    if ratio > 5:
        #print("variance")
        axis = max_axis
    else:
        #print("not variance")
        # Otherwise use cyclic axis
        axis = (parent_axis + 1) % 3

    node.axis = axis

    # -------------------------------------------------
    # Median split
    # -------------------------------------------------
    sorted_idx = np.argsort(points[:, axis])

    points = points[sorted_idx]
    indices = indices[sorted_idx]

    mid = len(points) // 2

    left_points = points[:mid]
    right_points = points[mid:]

    left_indices = indices[:mid]
    right_indices = indices[mid:]

    if len(left_points) > 0:
        node.left = _build_kdtree(
            left_points,
            left_indices,
            threshold,
            depth + 1,
            parent_axis=axis,
            pbar=pbar,
        )

    if len(right_points) > 0:
        node.right = _build_kdtree(
            right_points,
            right_indices,
            threshold,
            depth + 1,
            parent_axis=axis,
            pbar=pbar,
        )

    return node


# ---------------------------------------------------
# Collect leaf indices
# ---------------------------------------------------
def get_leaf_indices(node):
    leaves = []

    def _collect(n):
        if n.is_leaf():
            leaves.append(n.indices)
        else:
            if n.left is not None:
                _collect(n.left)
            if n.right is not None:
                _collect(n.right)

    _collect(node)
    return leaves


def round_robin(numpy_list):
    max_len = max(len(group) for group in numpy_list)

    chunks = []
    for i in range(max_len):
        chunk = []
        for group in numpy_list:
            chunk.append(group[i % len(group)])
        chunks.append(chunk)

    return chunks


def proposed_grouping(coord, threshold):
    points = np.asarray(coord)

    rough_max_nodes = 10000
    with tqdm(total=rough_max_nodes, desc="CPU KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)

    leaf_indices = get_leaf_indices(kdtree_root)

    return [np.array(t) for t in leaf_indices]


def create_chunks(coord, threshold):
    print("create chunks")

    numpy_list = proposed_grouping(coord, threshold)

    chunks = round_robin(numpy_list)

    chunks = [np.array(chunk) for chunk in chunks]

    return chunks, numpy_list