"""Barnes-Hut octree: build, compute centres of mass, and walk with an opening angle."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    centre: np.ndarray
    half: float
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    children: list = field(default_factory=list)
    body: int = -1

    @property
    def is_leaf(self) -> bool:
        return not self.children


def octant(centre: np.ndarray, point: np.ndarray) -> int:
    return int(point[0] > centre[0]) | int(point[1] > centre[1]) << 1 | int(point[2] > centre[2]) << 2


def child_centre(node: Node, index: int) -> np.ndarray:
    offset = np.array([(index >> axis) & 1 for axis in range(3)], dtype=float) * 2 - 1
    return node.centre + offset * node.half / 2


def insert(node: Node, index: int, pos: np.ndarray, depth: int = 0, max_depth: int = 48) -> None:
    if node.is_leaf and node.body == -1:
        node.body = index
        return
    if node.is_leaf:
        if depth >= max_depth:  # coincident bodies: keep them together in one leaf
            return
        node.children = [Node(child_centre(node, i), node.half / 2) for i in range(8)]
        existing, node.body = node.body, -1
        insert(node.children[octant(node.centre, pos[existing])], existing, pos, depth + 1)
    insert(node.children[octant(node.centre, pos[index])], index, pos, depth + 1)


def summarise(node: Node, pos: np.ndarray, mass: np.ndarray) -> None:
    if node.is_leaf:
        if node.body >= 0:
            node.mass, node.com = mass[node.body], pos[node.body].copy()
        return
    for child in node.children:
        summarise(child, pos, mass)
    node.mass = sum(child.mass for child in node.children)
    if node.mass > 0:
        node.com = sum(child.mass * child.com for child in node.children) / node.mass


def acceleration(node: Node, point: np.ndarray, theta: float, eps2: float) -> np.ndarray:
    if node.mass == 0:
        return np.zeros(3)
    delta = node.com - point
    dist2 = delta @ delta + eps2
    if node.is_leaf or (2 * node.half) ** 2 < theta * theta * dist2:
        return node.mass * delta / dist2 ** 1.5
    return sum(acceleration(child, point, theta, eps2) for child in node.children)


def build(pos: np.ndarray, mass: np.ndarray) -> Node:
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    root = Node(centre=(lo + hi) / 2, half=float((hi - lo).max()) / 2 + 1e-9)
    for index in range(len(pos)):
        insert(root, index, pos)
    summarise(root, pos, mass)
    return root
