import io
import zipfile
import json
from urllib.parse import unquote

import cv2
import numpy as np
import pikepdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docscan.api import main


@pytest.fixture
def client():
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def image_bytes():
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    cv2.putText(image, "Test page", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return cv2.imencode(".png", image)[1].tobytes()


PARAMS = {"align": "false", "trim": "false", "long_side": 600}


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "DocScanner" in response.text


@pytest.mark.parametrize("fmt,mime", [("png", "image/png"), ("jpg", "image/jpeg"), ("pdf", "application/pdf")])
def test_scan_formats_and_unicode_filename(client, image_bytes, fmt, mime):
    response = client.post("/scan", params={**PARAMS, "fmt": fmt},
                           files={"file": ("договор.png", image_bytes, "image/png")})
    assert response.status_code == 200
    assert response.headers["content-type"] == mime
    assert f"договор.{fmt}" in unquote(response.headers["content-disposition"])
    if fmt == "pdf":
        with pikepdf.open(io.BytesIO(response.content)) as pdf:
            assert len(pdf.pages) == 1
            assert float(pdf.pages[0].MediaBox[2]) == pytest.approx(600 / 300 * 72)
    else:
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
        assert image.shape == (450, 600)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_bad_upload_returns_400(client, data):
    response = client.post("/scan", files={"file": ("broken.png", data)})
    assert response.status_code == 400
    assert "broken.png" in response.json()["detail"]


def test_upload_byte_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_BYTES", 4)
    response = client.post("/scan", files={"file": ("large.png", b"12345")})
    assert response.status_code == 413


def test_upload_pixel_limit(client, image_bytes, monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/scan", files={"file": ("large.png", image_bytes)})
    assert response.status_code == 413


def test_transparent_editor_mask_becomes_white_paper():
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    image.putpixel((50, 50), (255, 0, 0, 255))
    data = io.BytesIO()
    image.save(data, "PNG")
    decoded = main._decode_image(data.getvalue(), "edited.png")
    np.testing.assert_array_equal(decoded[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(decoded[50, 50], [0, 0, 255])


def test_phone_exif_orientation_is_applied():
    image = Image.new("RGB", (400, 300), "white")
    exif = image.getexif()
    exif[274] = 6  # 90 degrees clockwise
    data = io.BytesIO()
    image.save(data, "JPEG", exif=exif)
    decoded = main._decode_image(data.getvalue(), "phone.jpg")
    assert decoded.shape == (400, 300, 3)


def test_zip_keeps_duplicate_names_and_order(client, image_bytes):
    names = ["page.png", "page.png", "page-2.png", "../PAGE.png", "C:\\folder\\page.png"]
    response = client.post("/scan/batch", params={**PARAMS, "out": "zip", "img_fmt": "png"},
                           files=[("files", (name, image_bytes, "image/png")) for name in names])
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["page.png", "page-2.png", "page-2-2.png", "PAGE-3.png", "page-4.png"]
        for data in map(archive.read, archive.namelist()):
            assert cv2.imdecode(np.frombuffer(data, np.uint8), 0).shape == (450, 600)


def test_batch_pdf_contains_every_page(client, image_bytes):
    response = client.post("/scan/batch", params=PARAMS,
                           files=[("files", (f"page-{n}.png", image_bytes)) for n in range(3)])
    assert response.status_code == 200
    with pikepdf.open(io.BytesIO(response.content)) as pdf:
        assert len(pdf.pages) == 3


def test_batch_does_not_silently_drop_bad_pages(client, image_bytes):
    response = client.post("/scan/batch", params=PARAMS, files=[
        ("files", ("good.png", image_bytes)), ("files", ("bad.png", b"bad")),
    ])
    assert response.status_code == 400
    assert "bad.png" in response.json()["detail"]


def test_batch_limit(client, image_bytes, monkeypatch):
    monkeypatch.setattr(main, "MAX_BATCH_FILES", 1)
    response = client.post("/scan/batch", files=[("files", ("page.png", image_bytes))] * 2)
    assert response.status_code == 413


def test_invalid_parameters(client, image_bytes):
    response = client.post("/scan?long_side=0", files={"file": ("page.png", image_bytes)})
    assert response.status_code == 422


def test_editor_crop_returns_exact_selected_pixels(client, image_bytes):
    response = client.post('/edit', data={'settings': json.dumps({'rotation': 90, 'crop': [30, 40, 100, 80]})},
                           files={'file': ('photo.png', image_bytes)})
    assert response.status_code == 200
    result = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    source = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(result, np.rot90(source, -1)[40:120, 30:130])


@pytest.mark.parametrize('settings', [
    'not JSON', '{"rotation":NaN}', '{"rotation":Infinity}', '{"rotation":361}',
    '{"quad":[[0,0]]}', '{"crop":[0,0,0,10]}', '{"unknown":true}',
    '{"quad":[[0,0],[400,300],[400,0],[0,300]]}',
])
def test_editor_bad_settings(client, image_bytes, settings):
    response = client.post('/edit', data={'settings': settings}, files={'file': ('photo.png', image_bytes)})
    assert response.status_code == 422


def test_editor_mask_and_geometry_together(client, image_bytes):
    mask = Image.new('RGBA', (400, 300), (0, 0, 0, 0))
    mask.paste((255, 255, 255, 255), (20, 20, 50, 50))
    data = io.BytesIO()
    mask.save(data, 'PNG')
    response = client.post('/edit', data={'settings': '{"crop":[20,20,30,30]}'}, files={
        'file': ('photo.png', image_bytes), 'mask': ('mask.png', data.getvalue()),
    })
    assert response.status_code == 200
    result = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)
    np.testing.assert_array_equal(result, np.full((30, 30, 3), 255, np.uint8))


def test_editor_invalid_mask_returns_error(client, image_bytes):
    response = client.post('/edit', data={'settings': '{}'}, files={
        'file': ('photo.png', image_bytes), 'mask': ('mask.png', image_bytes),
    })
    assert response.status_code == 400


def test_editor_rejects_output_that_exceeds_pixel_limit(client, image_bytes, monkeypatch):
    monkeypatch.setattr(main, 'MAX_IMAGE_PIXELS', 400*300)
    response = client.post('/edit', data={'settings': '{"rotation":45}'}, files={'file': ('photo.png', image_bytes)})
    assert response.status_code == 422
    assert 'Результат слишком большой' in response.json()['detail']


@pytest.mark.parametrize('mode', ['bw', 'gray', 'color'])
def test_preview_matches_saved_manual_edit_then_final_scan(client, image_bytes, mode):
    settings = json.dumps({'rotation': 90, 'crop': [30, 40, 100, 80]})
    preview = client.post('/preview', params={'mode': mode, 'long_side': 600, 'text_strength': .8},
                          data={'settings': settings}, files={'file': ('photo.png', image_bytes)})
    edited = client.post('/edit', data={'settings': settings}, files={'file': ('photo.png', image_bytes)})
    final = client.post('/scan', params={**PARAMS, 'mode': mode, 'text_strength': .8},
                        files={'file': ('edited.png', edited.content)})
    assert preview.status_code == edited.status_code == final.status_code == 200
    assert preview.content == final.content
    assert preview.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('strength', ['-1', '1.1', 'NaN', 'Infinity'])
@pytest.mark.parametrize('path', ['/preview', '/scan', '/scan/batch'])
def test_enhancement_strength_validation(client, image_bytes, strength, path):
    field = 'files' if path.endswith('batch') else 'file'
    response = client.post(path, params={'text_strength': strength}, files={field: ('photo.png', image_bytes)})
    assert response.status_code == 422


def test_detection_returns_editable_coordinates_without_discarding_pixels(client, image_bytes):
    response = client.post('/detect?align=false&trim=false&rotation=90', files={'file': ('photo.png', image_bytes)})
    assert response.status_code == 200
    data = response.json()
    assert data['kind'] == 'rect'
    np.testing.assert_allclose(data['crop'], [0, 0, 300, 400], atol=1e-5)
    assert data['found'] is False and data['trimmed'] is False


def test_detection_uses_same_conservative_defaults_as_scan(client, image_bytes, monkeypatch):
    def detect(image, **options):
        assert options['trim_margin_px'] == 0
        assert options['_trim_pad'] == 12
        return {'quad': [[20, 20], [380, 20], [380, 280], [20, 280]], 'found': True, 'trimmed': False}
    monkeypatch.setattr(main, 'suggest_crop', detect)
    response = client.post('/detect', files={'file': ('photo.png', image_bytes)})
    assert response.status_code == 200
    assert response.json()['kind'] == 'persp'


def test_detection_with_real_model_finds_the_paper_in_source_coordinates(client):
    image = np.full((600, 800, 3), 40, np.uint8)
    expected = np.array([[180, 45], [650, 95], [690, 550], [110, 500]], np.int32)
    cv2.fillConvexPoly(image, expected, (245, 245, 245))
    data = cv2.imencode('.png', image)[1].tobytes()
    response = client.post('/detect?trim=false', files={'file': ('photo.png', data)})
    assert response.status_code == 200
    detected = np.array(response.json()['quad'])
    assert response.json()['found']
    assert np.linalg.norm(detected - expected, axis=1).max() < 12


def test_preview_includes_brush_mask(client, image_bytes):
    mask = Image.new('RGBA', (400, 300), (0, 0, 0, 0))
    mask.paste((255, 255, 255, 255), (20, 20, 50, 50))
    data = io.BytesIO()
    mask.save(data, 'PNG')
    response = client.post('/preview?mode=color&text_strength=0&long_side=600',
        data={'settings': '{"crop":[20,20,30,30]}'}, files={
            'file': ('photo.png', image_bytes), 'mask': ('mask.png', data.getvalue()),
        })
    assert response.status_code == 200
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)
    assert image.shape == (600, 600, 3)
    assert image.min() == 255


def test_mixed_batch_preserves_manual_geometry_only_for_selected_pages(client, image_bytes, monkeypatch):
    calls = []
    def spy(image, **kwargs):
        calls.append((kwargs['align'], kwargs['trim']))
        return image
    monkeypatch.setattr(main, 'scan', spy)
    response = client.post('/scan/batch?out=zip&preserve=1', files=[
        ('files', ('auto.png', image_bytes)), ('files', ('manual.png', image_bytes)),
    ])
    assert response.status_code == 200
    assert calls == [(True, True), (False, False)]


def test_mixed_batch_rejects_invalid_manual_page_index(client, image_bytes):
    response = client.post('/scan/batch?preserve=2', files=[('files', ('page.png', image_bytes))])
    assert response.status_code == 422


@pytest.mark.parametrize('path', ['/static/editor.js', '/static/editor-geometry.js'])
def test_editor_modules_are_served(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert 'javascript' in response.headers['content-type']
