# tests/test_pipeline.py
import numpy as np
import cv2
import pytest

from tests._utils import synthetic_doc, is_mostly_white
from docscan.src import scan  # from .pipeline import scan в docscan/src/__init__.py

@pytest.mark.parametrize("mode", ["bw", "gray", "color"])
def test_scan_basic(mode):
    img = synthetic_doc()
    out = scan(
        img,
        output_mode=mode,
        target_long_side=1400,
        align=True,           # выравнивание перспективы
        trim=True,            # включить пост-трим
        trim_margin_px=6,     # внутренний отступ
        # без trim_thr/trim_sustain/trim_pad — их нет в сигнатуре scan()
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim in (2, 3)
    h, w = out.shape[:2]
    assert 900 <= max(h, w) <= 1500, "неожиданный размер результата"

    if mode == "bw":
        out_bgr = out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        assert is_mostly_white(out_bgr, min_ratio=0.55), "фон не похож на бумагу"
    elif mode == "gray":
        # Для полутонового результата проверим, что фон достаточно светлый в среднем.
        out_gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        # 0..255: пусть средняя яркость будет > 150 (достаточно «бумажно» даже с тенями)
        assert out_gray.mean() > 150, f"слишком тёмный фон в gray (mean={out_gray.mean():.1f})"

def test_scan_preserves_text_region():
    img = synthetic_doc()
    out = scan(
        img,
        output_mode="bw",
        target_long_side=1200,
        align=True,
        trim=True,
        trim_margin_px=6,
    )
    out_gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    white_ratio = (out_gray == 255).mean()
    assert white_ratio < 0.98, "слишком агрессивная бинаризация/обрезка"
