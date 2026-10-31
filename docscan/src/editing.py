"""Manual edits in image-edge coordinates, independent of the preview viewport."""

import cv2
import numpy as np


def rotation_layout(width: int, height: int, degrees: float):
    """Clockwise canvas rotation, mapping source edges to the rotated bounds."""
    radians = np.deg2rad(degrees)
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    out_width = max(1, int(np.ceil(abs(width * cosine) + abs(height * sine) - 1e-9)))
    out_height = max(1, int(np.ceil(abs(height * cosine) + abs(width * sine) - 1e-9)))
    matrix = np.array([
        [cosine, -sine, out_width / 2 - cosine * width / 2 + sine * height / 2],
        [sine, cosine, out_height / 2 - sine * width / 2 - cosine * height / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    return matrix, out_width, out_height


def apply_edit(image, *, rotation=0, crop=None, quad=None, mask=None, max_pixels=40_000_000):
    if not np.isfinite(rotation) or abs(rotation) > 360:
        raise ValueError("Недопустимый угол поворота")
    if crop is not None and quad is not None:
        raise ValueError("Выберите обрезку или четыре угла")
    height, width = image.shape[:2]
    transform, scene_width, scene_height = rotation_layout(width, height, rotation)
    out_width, out_height = scene_width, scene_height

    if crop is not None:
        values = np.asarray(crop, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Недопустимая область обрезки")
        x, y, w, h = values
        if min(x, y) < -1e-6 or min(w, h) < 1 or x + w > scene_width + 1e-6 or y + h > scene_height + 1e-6:
            raise ValueError("Область обрезки выходит за границы изображения")
        out_width, out_height = max(1, round(w)), max(1, round(h))
        crop_matrix = np.array([[out_width / w, 0, -x * out_width / w],
                                [0, out_height / h, -y * out_height / h], [0, 0, 1]])
        transform = crop_matrix @ transform
    elif quad is not None:
        points = np.asarray(quad, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Нужны четыре корректные угловые точки")
        if (points < -1e-6).any() or (points > [scene_width + 1e-6, scene_height + 1e-6]).any():
            raise ValueError("Углы выходят за границы изображения")
        edges = np.roll(points, -1, axis=0) - points
        following = np.roll(edges, -1, axis=0)
        crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        if not (crosses > 1e-6).all() or np.linalg.norm(edges, axis=1).min() < 2:
            raise ValueError("Углы должны идти по часовой стрелке и образовывать выпуклый четырёхугольник")
        out_width = max(1, round(max(np.linalg.norm(edges[0]), np.linalg.norm(edges[2]))))
        out_height = max(1, round(max(np.linalg.norm(edges[1]), np.linalg.norm(edges[3]))))
        destination = np.array([[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]], np.float32)
        transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination) @ transform

    if out_width * out_height > max_pixels:
        raise ValueError("Результат слишком большой; уменьшите область выделения")
    if mask is not None:
        if mask.shape != (height, width):
            raise ValueError("Размер маски не совпадает с изображением")
        image = image.copy()
        alpha = mask.astype(np.uint16)
        for channel in range(3):
            image[:, :, channel] = ((image[:, :, channel].astype(np.uint16) * (255 - alpha) + 255 * alpha) // 255).astype(np.uint8)

    # Canvas coordinates describe pixel edges; OpenCV transforms pixel centres.
    # This half-pixel conversion makes identity/crops/quarter turns exact.
    to_edges = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]])
    to_centres = np.array([[1, 0, -.5], [0, 1, -.5], [0, 0, 1]])
    return cv2.warpPerspective(image, to_centres @ transform @ to_edges, (out_width, out_height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
