import numpy as np
import cv2

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order a convex quad clockwise, starting at its top-left corner."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Expected four finite 2D points")
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 1:
        raise ValueError("Points must form a non-degenerate convex quadrilateral")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    # Unlike separate sum/difference extrema, this never duplicates a corner
    # when two corners have equal sums (e.g. a document rotated by 45 degrees).
    first = np.lexsort((ordered[:, 1], ordered.sum(axis=1)))[0]
    return np.roll(ordered, -first, axis=0)

def rotate_bound(image, angle_deg: float):
    # поворот без обрезки (для deskew)
    (h, w) = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC)

def four_point_transform(
    image,
    pts,
    min_side: int = 160,
    target_long_side: int | None = 2400,
    inter=cv2.INTER_CUBIC,
):
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    maxWidth = max(min_side, maxWidth)
    maxHeight = max(min_side, maxHeight)

    if target_long_side is not None:
        scale = float(target_long_side) / max(maxWidth, maxHeight)
        maxWidth = max(1, int(round(maxWidth * scale)))
        maxHeight = max(1, int(round(maxHeight * scale)))

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=inter)
    return warped
