import cv2
import numpy as np
import pytest

from docscan.src.editing import apply_edit, rotation_layout


@pytest.fixture
def pixels():
    return np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("angle,turns", [(0, 0), (90, -1), (-90, 1), (180, 2), (360, 0)])
def test_rotation_is_pixel_exact(pixels, angle, turns):
    np.testing.assert_array_equal(apply_edit(pixels, rotation=angle), np.rot90(pixels, turns))


def test_crop_is_pixel_exact_after_clockwise_rotation(pixels):
    result = apply_edit(pixels, rotation=90, crop=[5, 7, 20, 30])
    np.testing.assert_array_equal(result, np.rot90(pixels, -1)[7:37, 5:25])


def test_full_quad_does_not_implicitly_crop_or_rescale(pixels):
    result = apply_edit(pixels, quad=[[0, 0], [64, 0], [64, 48], [0, 48]])
    np.testing.assert_array_equal(result, pixels)


def test_real_projective_warp_restores_grid():
    grid = np.zeros((100, 150, 3), np.uint8)
    colors = np.random.default_rng(7).integers(30, 230, (4, 5, 3), dtype=np.uint8)
    for row in range(4):
        for col in range(5):
            grid[row*25:(row+1)*25, col*30:(col+1)*30] = colors[row, col]
    corners = np.array([[45, 20], [230, 50], [210, 185], [15, 160]], np.float32)
    source = np.array([[0, 0], [150, 0], [150, 100], [0, 100]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners)
    photo = cv2.warpPerspective(grid, matrix, (260, 210), borderValue=(255, 255, 255))
    result = apply_edit(photo, quad=corners)
    for row in range(4):
        for col in range(5):
            pixel = result[round((row+.5)*result.shape[0]/4), round((col+.5)*result.shape[1]/5)]
            np.testing.assert_allclose(pixel, colors[row, col], atol=2)


def test_brush_is_applied_in_source_coordinates_before_rotation(pixels):
    mask = np.zeros(pixels.shape[:2], np.uint8)
    mask[5:10, 15:20] = 255
    expected = pixels.copy()
    expected[5:10, 15:20] = 255
    np.testing.assert_array_equal(apply_edit(pixels, rotation=90, mask=mask), np.rot90(expected, -1))


def test_soft_brush_alpha_blends_onto_white():
    pixels = np.zeros((10, 10, 3), np.uint8)
    mask = np.full((10, 10), 128, np.uint8)
    np.testing.assert_array_equal(apply_edit(pixels, mask=mask), np.full_like(pixels, 128))


def test_arbitrary_rotation_has_white_padding():
    image = np.zeros((60, 100, 3), np.uint8)
    result = apply_edit(image, rotation=31)
    _, width, height = rotation_layout(100, 60, 31)
    assert result.shape == (height, width, 3)
    np.testing.assert_array_equal(result[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(result[height//2, width//2], [0, 0, 0])


@pytest.mark.parametrize("options", [
    {"crop": [-1, 0, 10, 10]}, {"crop": [0, 0, 65, 10]},
    {"crop": [0, 0, 0, 10]}, {"crop": [0, 0, float('inf'), 10]},
    {"quad": [[0, 0], [64, 48], [64, 0], [0, 48]]},
    {"quad": [[0, 0], [64, 0], [64, 0], [0, 48]]},
    {"quad": [[0, 0], [64, 0], [10, 10], [0, 48]]},
    {"quad": [[0, 0], [65, 0], [64, 48], [0, 48]]},
    {"crop": [0, 0, 20, 20], "quad": [[0, 0], [64, 0], [64, 48], [0, 48]]},
    {"rotation": float('nan')}, {"rotation": 361},
    {"mask": np.zeros((1, 1), np.uint8)}, {"max_pixels": 1},
])
def test_invalid_geometry_is_rejected(pixels, options):
    with pytest.raises(ValueError):
        apply_edit(pixels, **options)
