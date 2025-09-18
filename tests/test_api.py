# tests/test_api.py
import io
import zipfile
import cv2
import pytest

from tests._utils import synthetic_doc

def _to_file_tuple(img_bgr, name="img.jpg"):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return (name, io.BytesIO(buf.tobytes()), "image/jpeg")

@pytest.mark.parametrize("mode,fmt", [
    ("bw", "png"),
    ("gray", "jpg"),
    ("color", "pdf"),
])
def test_api_scan_single(client, mode, fmt):  # <-- просим фикстуру client аргументом
    img = synthetic_doc()
    files = {"file": _to_file_tuple(img, "one.jpg")}
    params = {
        "mode": mode,
        "long_side": 1200,
        "fmt": fmt,
        "align": "true",
        "trim": "true",
        "trim_margin": 6,
        "trim_thr": 0.22,
        "trim_sustain": 6,
        "trim_pad": 6,
    }
    r = client.post("/scan", params=params, files=files)
    assert r.status_code == 200, r.text
    ct = r.headers.get("content-type", "")
    if fmt in ("png", "jpg"):
        assert ct.startswith("image/")
    else:
        assert ct == "application/pdf"

def test_api_scan_batch_pdf_and_zip(client):  # <-- просим фикстуру client аргументом
    img1 = synthetic_doc()
    img2 = synthetic_doc(tilt=0.03)
    files = [
        ("files", _to_file_tuple(img1, "a.jpg")),
        ("files", _to_file_tuple(img2, "b.jpg")),
    ]
    params_pdf = {
        "mode": "bw", "long_side": 1100, "out": "pdf", "img_fmt": "jpg",
        "align": "true", "trim": "true"
    }
    r_pdf = client.post("/scan/batch", params=params_pdf, files=files)
    assert r_pdf.status_code == 200
    assert r_pdf.headers.get("content-type") == "application/pdf"

    params_zip = {
        "mode": "bw", "long_side": 1100, "out": "zip", "img_fmt": "jpg",
        "align": "true", "trim": "true"
    }
    r_zip = client.post("/scan/batch", params=params_zip, files=files)
    assert r_zip.status_code == 200
    assert r_zip.headers.get("content-type") == "application/zip"

    # zip валиден и содержит 2 файла
    z = zipfile.ZipFile(io.BytesIO(r_zip.content))
    assert len(z.namelist()) == 2
