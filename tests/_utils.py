# tests/_utils.py
from __future__ import annotations
import cv2
import numpy as np

def synthetic_doc(
    w: int = 1280,
    h: int = 960,
    tilt: float = 0.08,
    noise: float = 8.0,
    shadow: bool = True,
) -> np.ndarray:
    """
    Возвращает BGR-картинку, имитирующую документ на столе:
    - коричневый фон
    - белый лист с небольшим perspective warp
    - немного шума и тени
    """
    rng = np.random.default_rng(123)
    # фон
    bg = np.full((h, w, 3), (50, 80, 110), np.uint8)  # «стол»
    cv2.randn(bg, (60, 85, 115), (8, 8, 8))          # шум по каналу

    # белый прямоугольник (лист) центрирован
    pad = int(min(w, h) * 0.12)
    x0, y0 = pad, pad
    x1, y1 = w - pad, h - pad
    doc = bg.copy()
    cv2.rectangle(doc, (x0, y0), (x1, y1), (245, 245, 245), -1)

    # лёгкая перспектива
    dx = int(tilt * (x1 - x0))
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    dst = np.float32([[x0+dx, y0], [x1-dx, y0+dx//3], [x1- dx, y1-dx//2], [x0+dx//2, y1-dx]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # лёгкая тень сверху
    if shadow:
        mask = np.linspace(0.85, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
        warped = (warped.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

    # шум
    if noise > 0:
        n = rng.normal(0, noise, size=warped.shape).astype(np.float32)
        warped = np.clip(warped.astype(np.float32) + n, 0, 255).astype(np.uint8)

    return warped

def is_mostly_white(img_bgr: np.ndarray, min_ratio: float = 0.75) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    ratio = th.mean() / 255.0
    return ratio >= min_ratio
