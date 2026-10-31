from itertools import permutations

import cv2
import numpy as np
import pytest

from docscan.src.geometry import four_point_transform, order_points


def test_rotated_document_keeps_four_distinct_corners():
    points = [[50, 0], [100, 50], [50, 100], [0, 50]]
    for permutation in permutations(points):
        ordered = order_points(permutation)
        np.testing.assert_array_equal(ordered, points)


@pytest.mark.parametrize("points", [
    [[0, 0], [0, 0], [100, 100], [0, 100]],
    [[0, 0], [50, 50], [100, 100], [0, 100]],
    [[0, 0], [100, 0], [float('nan'), 100], [0, 100]],
])
def test_invalid_quad_is_rejected(points):
    with pytest.raises(ValueError):
        order_points(points)


@pytest.mark.parametrize("target", [100, 800])
def test_perspective_transform_resizes_both_directions(target):
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    points = np.array([[0, 0], [599, 0], [599, 399], [0, 399]], dtype=np.float32)
    result = four_point_transform(image, points, target_long_side=target)
    assert max(result.shape[:2]) == target
    assert abs(result.shape[1] / result.shape[0] - 1.5) < 0.03


def test_diamond_transform_preserves_content():
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    points = np.array([[110, 10], [210, 110], [110, 210], [10, 110]], dtype=np.float32)
    cv2.fillConvexPoly(image, points.astype(np.int32), (255, 255, 255))
    result = four_point_transform(image, points, target_long_side=300)
    assert result.shape == (300, 300, 3)
    assert result.mean() > 250
