import cv2
import numpy as np
import pytest

from docscan.src import pipeline
from docscan.src import scan
from tests._utils import synthetic_doc, is_mostly_white


@pytest.mark.parametrize("mode,channels", [("bw", 2), ("gray", 2), ("color", 3)])
@pytest.mark.parametrize("target", [600, 1000, None])
def test_resize_without_alignment(mode, channels, target, monkeypatch):
    def no_hed(*args):
        pytest.fail("HED must not be loaded when alignment is disabled")

    monkeypatch.setattr(pipeline, "HED", no_hed)
    image = np.full((400, 800, 3), 240, dtype=np.uint8)
    cv2.putText(image, "Document", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    result = pipeline.scan(image, output_mode=mode, target_long_side=target, align=False, trim=False)
    assert result.ndim == channels
    assert max(result.shape[:2]) == (target or 800)
    assert result.shape[1] == 2 * result.shape[0]
    if mode == "bw":
        assert set(np.unique(result)).issubset({0, 255})


def test_no_contour_fallback_honours_size(monkeypatch):
    class NoHED:
        available = False

        def __init__(self, *args):
            pass

    monkeypatch.setattr(pipeline, "HED", NoHED)
    monkeypatch.setattr(pipeline, "_find_doc_quad", lambda image, hed: (None, image))
    image = np.full((400, 800, 3), 255, dtype=np.uint8)
    result = pipeline.scan(image, target_long_side=600, trim=False)
    assert result.shape == (300, 600)


def test_size_is_applied_after_trim(monkeypatch):
    monkeypatch.setattr(pipeline, "_auto_trim_to_mask_kmeans", lambda image, **kwargs: image[10:-10, 10:-10])
    image = np.full((400, 800, 3), 255, dtype=np.uint8)
    result = pipeline.scan(image, target_long_side=600, align=False, trim=True)
    assert max(result.shape[:2]) == 600


@pytest.mark.parametrize('mode', ['gray', 'color'])
def test_text_enhancement_can_be_completely_disabled(mode):
    image = np.random.default_rng(42).integers(0, 256, (100, 150, 3), np.uint8)
    result = pipeline.scan(image, output_mode=mode, target_long_side=None, align=False, trim=False, text_strength=0)
    expected = image if mode == 'color' else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(result, expected)


def test_enhancer_makes_blurred_text_edges_sharper_without_background_noise():
    image = np.full((300, 700), 235, np.uint8)
    cv2.putText(image, 'Invoice 123', (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 130, 2)
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    strengths = [pipeline._text_enhance(image, value) for value in [0, .5, 1]]
    sharpness = [cv2.Laplacian(result, cv2.CV_64F).var() for result in strengths]
    assert sharpness[0] < sharpness[1] < sharpness[2]
    assert strengths[-1][220:, :].std() < 2


@pytest.mark.parametrize('strength', [-1, 1.1, float('nan'), float('inf')])
def test_enhancer_rejects_invalid_strength(strength):
    with pytest.raises(ValueError):
        pipeline._text_enhance(np.full((20, 20), 200, np.uint8), strength)


def test_crop_suggestion_keeps_source_and_maps_trim_to_original_pixels(monkeypatch):
    source = np.full((200, 300, 3), 240, np.uint8)
    source[0:10] = (0, 0, 255)
    original = source.copy()
    monkeypatch.setattr(pipeline, '_find_trim_rect', lambda image, **kwargs: (10, 20, 280, 180))
    result = pipeline.suggest_crop(source, align=False)
    np.testing.assert_allclose(result['quad'], [[10, 20], [280, 20], [280, 180], [10, 180]], atol=1e-5)
    np.testing.assert_array_equal(source, original)
    assert result['trimmed'] is True


def test_crop_suggestion_maps_trim_back_through_perspective(monkeypatch):
    source = np.full((600, 800, 3), 240, np.uint8)
    quad = np.array([[180, 45], [650, 95], [690, 550], [110, 500]], np.float32)
    monkeypatch.setattr(pipeline, '_detect_source_quad', lambda image: (quad, image))
    monkeypatch.setattr(pipeline, '_find_trim_rect', lambda image, **kwargs: (10, 20, image.shape[1]-10, image.shape[0]-20))
    result = pipeline.suggest_crop(source)
    assert result['found'] and result['trimmed']
    detected = np.array(result['quad'], np.float32)
    for point in detected:
        assert cv2.pointPolygonTest(quad, tuple(map(float, point)), False) == 1


def test_safe_trim_defaults_keep_more_pixels_than_previous_defaults(monkeypatch):
    # A dark band around white paper; deterministic mask avoids k-means randomness.
    image = np.full((200, 300, 3), 240, np.uint8)
    mask = np.full((200, 300), 255, np.uint8)
    mask[10:190, 10:290] = 0
    monkeypatch.setattr(pipeline, '_kmeans_bg_mask', lambda image: (mask, None))
    current = pipeline._find_trim_rect(image)
    previous = pipeline._find_trim_rect(image, margin_px=8, pad_px=6, max_shrink_ratio=.25)
    assert current[0] <= previous[0] and current[1] <= previous[1]
    assert current[2] >= previous[2] and current[3] >= previous[3]


def test_trim_full_frame_does_not_lose_last_pixels_after_downscaling(monkeypatch):
    image = np.full((1200, 1600, 3), 240, np.uint8)
    mask = np.zeros((675, 900), np.uint8)
    monkeypatch.setattr(pipeline, '_kmeans_bg_mask', lambda image: (mask, None))
    assert pipeline._find_trim_rect(image) == (0, 0, 1600, 1200)


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
