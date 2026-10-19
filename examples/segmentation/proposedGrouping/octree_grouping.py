import numpy as np


# =====================================================
# PARAMETERS
# =====================================================

MAX_DEPTH = 30
MIN_CELL_SIZE = 1e-6


# =====================================================
# NODE
# =====================================================

class OctreeNode:
    def __init__(self, indices):
        self.indices = indices
        self.children = [None] * 8

    def is_leaf(self):
        return all(child is None for child in self.children)


# =====================================================
# OCTREE BUILDING
# =====================================================

def build_octree(points, threshold):

    indices = np.arange(points.shape[0])

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)

    return _build_octree(
        points,
        indices,
        bbox_min,
        bbox_max,
        threshold,
        depth=0
    )


def _build_octree(
    points,
    indices,
    bbox_min,
    bbox_max,
    threshold,
    depth=0
):

    node = OctreeNode(indices)

    # Stop 1: leaf size reached
    if len(points) <= threshold:
        return node

    # Stop 2: depth limit
    if depth >= MAX_DEPTH:
        return node

    # Stop 3: cell too small
    if np.max(bbox_max - bbox_min) < MIN_CELL_SIZE:
        return node

    center = (bbox_min + bbox_max) * 0.5

    x_bit = (points[:, 0] >= center[0]).astype(np.int8)
    y_bit = (points[:, 1] >= center[1]).astype(np.int8)
    z_bit = (points[:, 2] >= center[2]).astype(np.int8)

    octant_ids = x_bit + (y_bit << 1) + (z_bit << 2)

    for octant in range(8):

        mask = octant_ids == octant

        if not np.any(mask):
            continue

        child_points = points[mask]
        child_indices = indices[mask]

        child_min = bbox_min.copy()
        child_max = bbox_max.copy()

        if octant & 1:
            child_min[0] = center[0]
        else:
            child_max[0] = center[0]

        if octant & 2:
            child_min[1] = center[1]
        else:
            child_max[1] = center[1]

        if octant & 4:
            child_min[2] = center[2]
        else:
            child_max[2] = center[2]

        node.children[octant] = _build_octree(
            child_points,
            child_indices,
            child_min,
            child_max,
            threshold,
            depth + 1
        )

    return node


# =====================================================
# LEAF COLLECTION
# =====================================================

def get_leaf_indices(node):

    leaves = []

    def _collect(n):
        if n.is_leaf():
            leaves.append(n.indices)
            return

        for child in n.children:
            if child is not None:
                _collect(child)

    _collect(node)

    return leaves


# =====================================================
# ROUND ROBIN
# =====================================================

def round_robin(numpy_list):

    max_len = max(len(group) for group in numpy_list)

    chunks = []

    for i in range(max_len):
        chunks.append([
            group[i % len(group)]
            for group in numpy_list
        ])

    return chunks


# =====================================================
# GROUPING
# =====================================================

def proposed_grouping(coord, threshold):

    points = np.asarray(coord)

    octree_root = build_octree(
        points,
        threshold
    )

    leaf_indices = get_leaf_indices(octree_root)

    return [np.asarray(t) for t in leaf_indices]


# =====================================================
# CREATE CHUNKS
# =====================================================

def create_chunks(coord, threshold):

    numpy_list = proposed_grouping(
        coord,
        threshold
    )

    chunks = round_robin(numpy_list)

    chunks = [np.asarray(chunk) for chunk in chunks]

    return chunks, numpy_list