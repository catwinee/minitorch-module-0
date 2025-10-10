import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Randomly generate N 2D points.

    Args:
        N: The number of the points to generate.

    """
    X = []
    for _ in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Create a simple binary classification dataset.

    Points with an x_1 coordinate less than 0.5 are assigned a positive
    label (1), and a negative label (0) otherwise.

    Args:
        N: The total number of data points to generate.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Create a dataset with a diagonal decision boundary.

    Points with sum of the coordinates x_1 + x_2 less than 0.5 are
    assigned a positive label (1), and a negative label (0) otherwise.

    Args:
        N: The total number of data points to generate.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Create a dataset with a vertical split decision boundary.

    Points are assigned a positive label (1) if their x_1 coordinate
    is less than 0.2 or greater than 0.8.

    Args:
        N: The total number of data points to generate.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Create an XOR-like dataset.

    Points in the top-left and bottom-right quadrants
    are assigned a positive label (1).

    Args:
        N: The total number of data points to generate.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Create a dataset with a circular decision boundary.

    Points outside a circle centered at (0.5, 0.5) are assigned a
    positive label (1).

    Args:
        N: The total number of data points to generate.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Create a dataset with two intertwined spirals.

    Each spiral belongs to a different class.

    Args:
        N: The total number of data points to generate.
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
