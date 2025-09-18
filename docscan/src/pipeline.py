from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from .geometry import four_point_transform, order_points, rotate_bound
from .hed import HED

# ================== БАЗОВЫЕ ФИЛЬТРЫ ==================

def _gray_world_wb(img_bgr):
    img = img_bgr.astype(np.float32)
    mean = img.reshape(-1,3).mean(axis=0)
    g = mean.mean()
    gain = g / (mean + 1e-6)
    img *= gain
    return np.clip(img, 0, 255).astype(np.uint8)

def _auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def _sharpen(image, strength=0.55, radius=1.2):
    blur = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def _text_enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = _sharpen(g, strength=0.25, radius=1.0)
    return g

# ================== ВСПОМОГАТЕЛЬНОЕ ==================

def _rect_score(pts):
    r = order_points(pts.astype("float32"))
    v1 = r[1]-r[0]; v2 = r[2]-r[1]; v3 = r[3]-r[2]; v4 = r[0]-r[3]
    def ang(a,b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na*nb == 0: return 0
        cs = np.clip((a@b)/(na*nb), -1, 1)
        return np.degrees(np.arccos(cs))
    a1 = abs(ang(v1, v2) - 90)
    a2 = abs(ang(v2, v3) - 90)
    a3 = abs(ang(v3, v4) - 90)
    a4 = abs(ang(v4, v1) - 90)
    # чем ближе к прямоугольнику и больше площадь — тем лучше
    return - (a1+a2+a3+a4)

def _is_valid_quad(approx, img_area, min_area_ratio=0.08, max_area_ratio=0.98,
                   min_aspect=0.3, max_aspect=3.8, min_side_px=140):
    if len(approx) != 4: return False
    area = abs(cv2.contourArea(approx))
    if area <= 0: return False
    area_ratio = area / float(img_area)
    if not (min_area_ratio <= area_ratio <= max_area_ratio): return False
    if not cv2.isContourConvex(approx): return False
    # предикция размеров по 4 точкам
    r = order_points(approx.reshape(4,2).astype("float32"))
    wA = np.linalg.norm(r[2] - r[3]); wB = np.linalg.norm(r[1] - r[0])
    hA = np.linalg.norm(r[1] - r[2]); hB = np.linalg.norm(r[0] - r[3])
    w = max(wA, wB); h = max(hA, hB)
    if w < min_side_px or h < min_side_px: return False
    asp = w / max(h, 1e-6)
    if not (min_aspect <= asp <= max_aspect): return False
    return True

# ================== KMEANS ФОН/БУМАГА ==================

def _kmeans_bg_mask(bgr_small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Быстрый k-means (k=2) в LAB по краевым полосам, чтобы различить ФОН vs БУМАГА.
    Возвращает (bg_mask 0/255, centers[2,3] в LAB).
    """
    h, w = bgr_small.shape[:2]
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = max(4, int(round(0.06 * min(h, w))))
    top  = lab[:border, :, :].reshape(-1,3)
    bot  = lab[h-border:, :, :].reshape(-1,3)
    lef  = lab[:, :border, :].reshape(-1,3)
    rig  = lab[:, w-border:, :].reshape(-1,3)
    samples = np.vstack([top, bot, lef, rig])
    if samples.shape[0] > 200_000:
        idx = np.random.choice(samples.shape[0], 200_000, replace=False)
        samples = samples[idx]
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _comp, _labels, centers = cv2.kmeans(samples, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    bg_idx = int(np.argmin(centers[:, 0]))  # более тёмный по L* — фон
    L = lab[:,:,0]; A = lab[:,:,1]; B = lab[:,:,2]
    c_bg  = centers[bg_idx]; c_oth = centers[1-bg_idx]
    d_bg  = (L - c_bg[0])**2 + (A - c_bg[1])**2 + (B - c_bg[2])**2
    d_oth = (L - c_oth[0])**2 + (A - c_oth[1])**2 + (B - c_oth[2])**2
    bg_mask = (d_bg <= d_oth).astype(np.uint8) * 255
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 1)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)), 1)
    bg_mask = cv2.medianBlur(bg_mask, 5)
    return bg_mask, centers

# ================== ПОИСК КОНТУРА ДОКУМЕНТА ==================

def _find_doc_quad(small_bgr: np.ndarray, hed: HED | None) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Пытаемся найти 4-угольник документа.
    1) Canny (+HED)
    2) fallback: из маски бумаги (по k-means) берём самый крупный контур и упрощаем до 4 углов
    Возвращает (approx_quad или None, small_bgr_used)
    """
    h, w = small_bgr.shape[:2]
    img_area = h * w

    # слегка подавим «стол», чтобы бумага выступала
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]; v = hsv[:,:,2]
    paper_like = (s < 70) & (v > 160)
    atten = cv2.blur((~paper_like).astype(np.uint8) * 255, (9,9)).astype(np.float32)/255.0
    bgr_att = (small_bgr.astype(np.float32) * (1.0 - 0.25*atten[...,None])).astype(np.uint8)

    gray = cv2.cvtColor(bgr_att, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = _auto_canny(gray_blur, sigma=0.33)

    if hed is not None and hed.available:
        hed_map = hed.predict(bgr_att)
        if hed_map is not None:
            hed_bin = cv2.threshold(hed_map, 40, 255, cv2.THRESH_BINARY)[1]
            hed_bin = cv2.GaussianBlur(hed_bin, (3,3), 0)
            mix = cv2.addWeighted(canny, 0.7, hed_bin, 0.9, 0)
            _, edges = cv2.threshold(mix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            edges = canny
    else:
        edges = canny

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)),1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = -1e9
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:20]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if _is_valid_quad(approx, img_area):
            score = _rect_score(approx.reshape(4,2)) + 0.0002 * cv2.contourArea(approx)
            if score > best_score:
                best, best_score = approx, score
            continue
        # попробуем minAreaRect, если approx не дал 4 угла
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32).reshape(-1,1,2)
        if _is_valid_quad(box, img_area):
            score = _rect_score(box.reshape(4,2)) + 0.00015 * cv2.contourArea(c)
            if score > best_score:
                best, best_score = box, score

    if best is not None:
        return best, bgr_att

    # ---- fallback: из маски бумаги (k-means) взять крупнейший компонент
    bg_mask, _ = _kmeans_bg_mask(bgr_att)
    paper = cv2.bitwise_not(bg_mask)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), 1)
    cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, bgr_att
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4 and _is_valid_quad(approx, img_area, min_area_ratio=0.06):
        return approx, bgr_att
    # если вершин больше — возьмём minAreaRect как стабильный прямоугольник
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32).reshape(-1,1,2)
    if _is_valid_quad(box, img_area, min_area_ratio=0.06):
        return box, bgr_att
    return None, bgr_att

# ================== ТРИМ ПО K-MEANS (С ПАРАМЕТРАМИ) ==================

def _auto_trim_to_mask_kmeans(
    bgr,
    margin_px=6,
    pad_px=6,
    max_shrink_ratio=0.25,
    thr=0.22,
    sustain=6,
):
    """
    Трим по цвету границ (k-means на LAB по периметру), параметризованный.
    thr: порог доли ФОНА (меньше — бумага началась).
    sustain: скольких подряд строк/столбцов достаточно для уверенности.
    """
    H, W = bgr.shape[:2]
    if H < 80 or W < 80:
        return bgr

    scale = 900.0 / max(H, W)
    if scale < 1.0:
        small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr.copy()
        scale = 1.0
    h, w = small.shape[:2]

    bg_mask, _ = _kmeans_bg_mask(small)
    if bg_mask.mean() > 225:  # фон "захватил всё" — сомнительно
        return bgr

    row_bg = (bg_mask.mean(axis=1) / 255.0).astype(np.float32)
    col_bg = (bg_mask.mean(axis=0) / 255.0).astype(np.float32)
    k = 9
    row_bg = cv2.GaussianBlur(row_bg.reshape(-1,1), (1,k), 0).ravel()
    col_bg = cv2.GaussianBlur(col_bg.reshape(1,-1), (k,1), 0).ravel()

    sust_rows = max(3, int(sustain))
    sust_cols = max(3, int(sustain))

    # top
    y0 = 0; run = 0
    for y in range(h):
        if row_bg[y] < thr:
            run += 1
            if run >= sust_rows:
                y0 = max(0, y - sust_rows + 1); break
        else: run = 0
    # bottom
    y1 = h-1; run = 0
    for y in range(h-1, -1, -1):
        if row_bg[y] < thr:
            run += 1
            if run >= sust_rows:
                y1 = min(h-1, y + sust_rows - 1); break
        else: run = 0
    # left
    x0 = 0; run = 0
    for x in range(w):
        if col_bg[x] < thr:
            run += 1
            if run >= sust_cols:
                x0 = max(0, x - sust_cols + 1); break
        else: run = 0
    # right
    x1 = w-1; run = 0
    for x in range(w-1, -1, -1):
        if col_bg[x] < thr:
            run += 1
            if run >= sust_cols:
                x1 = min(w-1, x + sust_cols - 1); break
        else: run = 0

    m = int(round(margin_px * scale))
    x0 = max(0, x0 + m); y0 = max(0, y0 + m)
    x1 = min(w-1, x1 - m); y1 = min(h-1, y1 - m)

    pad = max(0, int(round(pad_px * scale)))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w-1, x1 + pad); y1 = min(h-1, y1 + pad)

    if x1 - x0 < 30 or y1 - y0 < 30:
        return bgr

    X0 = int(round(x0 / scale)); Y0 = int(round(y0 / scale))
    X1 = int(round(x1 / scale)); Y1 = int(round(y1 / scale))
    newW = X1 - X0 + 1; newH = Y1 - Y0 + 1
    if newW < 50 or newH < 50:
        return bgr
    if (W - newW)/W > max_shrink_ratio or (H - newH)/H > max_shrink_ratio:
        return bgr
    return bgr[Y0:Y1+1, X0:X1+1].copy()

# ================== ОСНОВНОЙ PIPELINE ==================

def scan(
    image_bgr: np.ndarray,
    output_mode: str = "bw",
    debug: bool = False,
    target_long_side: int | None = 2400,
    # выравнивание
    align: bool = True,
    # трим полей
    trim: bool = True,
    trim_margin_px: int = 8,
    _trim_thr: float = 0.22,
    _trim_sustain: int = 6,
    _trim_pad: int = 6,
) -> np.ndarray:
    """
    1) Цвет/шум нормализация
    2) Поиск 4-уг. документа (Canny+HED, fallback k-means) и ПЕРСПЕКТИВНОЕ ВЫРАВНИВАНИЕ
    3) Трим по k-means (параметризован)
    4) Режимы вывода: bw | gray | color (+ enhancement)
    """
    # 1) нормализация
    img = _gray_world_wb(image_bgr)
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

    H, W = img.shape[:2]
    scale = 1300.0 / max(H, W)
    small = cv2.resize(img, (int(W*scale), int(H*scale))) if scale < 1.0 else img.copy()
    ratio = image_bgr.shape[0] / float(small.shape[0])

    # 2) выравнивание (перспективное)
    doc = image_bgr.copy()
    hed = HED(Path(__file__).resolve().parent.parent / "models" / "hed")
    if align:
        quad, small_used = _find_doc_quad(small, hed if hed.available else None)
        if quad is not None:
            pts = quad.reshape(4,2) * ratio
            doc = four_point_transform(
                image_bgr, pts,
                min_side=260,
                target_long_side=target_long_side,
                inter=cv2.INTER_CUBIC
            )
        else:
            # нет надёжного 4-угольника — просто бережно ресайзим
            if target_long_side is not None:
                h0, w0 = doc.shape[:2]
                s = target_long_side / max(h0, w0)
                if s > 1.0:
                    doc = cv2.resize(doc, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_CUBIC)

    # 3) трим по k-means
    if trim:
        doc = _auto_trim_to_mask_kmeans(
            doc,
            margin_px=trim_margin_px,
            pad_px=_trim_pad,
            max_shrink_ratio=0.25,
            thr=float(_trim_thr),
            sustain=int(_trim_sustain),
        )

    # 4) улучшение читаемости
    if output_mode == "color":
        out = _sharpen(doc, strength=0.35, radius=1.2)
    else:
        g = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
        g = _text_enhance(g)
        if output_mode == "gray":
            out = g
        else:
            out = cv2.adaptiveThreshold(
                g, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 35, 10
            )

    if debug:
        return out, {"small": small}
    return out
