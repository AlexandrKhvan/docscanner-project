# tests/conftest.py
import os
import io
import cv2
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Если у вас в pipeline есть опциональный HED / DL, можно выключить его на CI:
os.environ.setdefault("DOCSCAN_DISABLE_HED", "1")

from docscan.api.main import app  # FastAPI приложение
from tests._utils import synthetic_doc

@pytest.fixture(scope="session")
def test_image_bgr() -> np.ndarray:
    return synthetic_doc(w=1280, h=960, tilt=0.07, noise=6.0, shadow=True)

@pytest.fixture(scope="session")
def test_image_bytes(test_image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", test_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, "cv2.imencode failed"
    return buf.tobytes()

@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)

def pil_bytes(img_bgr: np.ndarray, fmt="PNG") -> bytes:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bio = io.BytesIO()
    Image.fromarray(rgb).save(bio, format=fmt)
    return bio.getvalue()
