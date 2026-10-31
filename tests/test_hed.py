from pathlib import Path

import cv2
import numpy as np
import pytest

from docscan.src.geometry import order_points
from docscan.src.hed import HED
from docscan.src.pipeline import _find_doc_quad


def test_missing_model_has_contour_fallback(tmp_path):
    hed = HED(tmp_path)
    assert not hed.available
    assert hed.predict(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_real_hed_finds_document_without_shifting_its_edges():
    models = Path(__file__).resolve().parents[1] / "docscan" / "models" / "hed"
    if not (models / "hed_pretrained_bsds.caffemodel").is_file():
        pytest.skip("Optional HED weights are not installed")
    hed = HED(models)
    assert hed.available
    image = np.full((600, 800, 3), 40, dtype=np.uint8)
    corners = np.array([[180, 45], [650, 95], [690, 550], [110, 500]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (245, 245, 245))
    detected, _ = _find_doc_quad(image, hed)
    assert detected is not None
    distances = np.linalg.norm(order_points(detected.reshape(4, 2)) - order_points(corners), axis=1)
    assert distances.max() < 12, distances
