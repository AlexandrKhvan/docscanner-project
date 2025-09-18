import cv2
import numpy as np
from pathlib import Path
from .geometry import four_point_transform, order_points, rotate_bound
from .hed import HED

# ---------- ВСПОМОГАТЕЛЬНОЕ ----------

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

# ---------- МАСКА БУМАГИ И АВТО-ТРИМ ----------

def _paper_mask(bgr):
    """
    Маска «бумаги»: светлая, малонасыщенная, близкая к белому.
    Возвращает uint8 0/255.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # динамические пороги по распределению яркости (защита от тёмных/жёлтых сцен)
    v_thr = np.percentile(v, 70)
    L_thr = np.percentile(L, 65)

    bright = (v >= max(180, v_thr)).astype(np.uint8) * 255
    light  = (L >= max(170, L_thr)).astype(np.uint8) * 255
    low_sat = (s <= 60).astype(np.uint8) * 255

    # близко к белому по a,b: считаем на int16, чтобы не было переполнения
    a_dev = np.abs(A.astype(np.int16) - 128)
    b_dev = np.abs(B.astype(np.int16) - 128)
    near_white_ab = ((a_dev <= 18) & (b_dev <= 18)).astype(np.uint8) * 255

    m = cv2.bitwise_and(bright, light)
    m = cv2.bitwise_and(m, low_sat)
    m = cv2.bitwise_and(m, near_white_ab)

    # морфология: закрыть дырки, сгладить
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)
    return m

def _edge_density(gray, x0, y0, x1, y1):
    x0 = max(0, min(x0, gray.shape[1])); x1 = max(0, min(x1, gray.shape[1]))
    y0 = max(0, min(y0, gray.shape[0])); y1 = max(0, min(y1, gray.shape[0]))
    if x1 <= x0 or y1 <= y0: return 0.0
    roi = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3,3), 0)
    e = cv2.Canny(roi, 60, 180)
    return float(e.mean()) / 255.0

def _smooth_1d(arr, k=9):
    k = int(max(3, k | 1))
    if arr.ndim == 1:
        tmp = cv2.GaussianBlur(arr.reshape(1, -1).astype(np.float32), (k,1), 0).ravel()
        return tmp
    return arr

def _first_run_from(arr, start, step, thr, sustain):
    n = len(arr); i = start; ok = 0
    while 0 <= i < n:
        if arr[i] >= thr:
            ok += 1
            if ok >= sustain:
                return max(0, min(n-1, i - sustain + 1))
        else:
            ok = 0
        i += step
    return start

def _find_edge_by_gradient(arr, direction, sustain, thr, search_frac=0.65):
    """
    Ищем резкий фронт 'стол -> бумага' по 1D-градиенту.
    direction: 'top'|'bottom'
    Возвращает индекс границы (y для top/bottom).
    """
    n = len(arr)
    sm = _smooth_1d(arr, k=9)
    g = np.gradient(sm)
    if direction == 'top':
        # ищем максимум положительного градиента в верхней части кадра
        upto = max(10, int(search_frac * n))
        idx = int(np.argmax(g[:upto]))
        # валидация: после границы sustain строк должны быть "бумагой"
        if idx + sustain < n and np.mean(arr[idx+1:idx+1+sustain]) >= thr:
            return idx
        # fallback: порог по sustain
        return _first_run_from(arr, 0, +1, thr, sustain)
    else:  # 'bottom'
        # ищем максимум отрицательного градиента в нижней части (или минимум g)
        from_ = int((1.0 - search_frac) * n)
        idx = int(np.argmin(g[from_:])) + from_
        if idx - sustain >= 0 and np.mean(arr[idx-sustain:idx]) >= thr:
            return idx
        # fallback: порог по sustain (с конца)
        return _first_run_from(arr, n-1, -1, thr, sustain)

def _auto_trim_to_mask(
    bgr,
    margin_px=6,
    min_keep_ratio=0.5,
    pad_px=6,
    max_shrink_ratio=0.25
):
    """
    Трим по цвету границ (k-means на LAB пикселях из окантовки кадра).
    1) downscale для скорости
    2) k-means (k=2) на пикселях узких полос по периметру -> фон/бумага
    3) строим маску ФОНА по ближайшему центру
    4) проекции доли фона по строкам/столбцам -> ищем, где фон почти исчезает
    5) применяем в исходном масштабе; есть предохранитель по max_shrink_ratio
    """
    H, W = bgr.shape[:2]
    if H < 80 or W < 80:
        return bgr

    # ---- 1) даунскейл
    scale = 900.0 / max(H, W)
    if scale < 1.0:
        small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr.copy()
        scale = 1.0
    h, w = small.shape[:2]

    # ---- 2) k-means по LAB пикселям из окантовки
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = max(4, int(round(0.06 * min(h, w))))
    top_strip  = lab[:border, :, :].reshape(-1, 3)
    bot_strip  = lab[h-border:, :, :].reshape(-1, 3)
    left_strip = lab[:, :border, :].reshape(-1, 3)
    right_strip= lab[:, w-border:, :].reshape(-1, 3)
    samples = np.vstack([top_strip, bot_strip, left_strip, right_strip])

    # подстрахуемся: если очень много семплов, возьмём случайную подвыборку
    if samples.shape[0] > 200_000:
        idx = np.random.choice(samples.shape[0], 200_000, replace=False)
        samples = samples[idx]

    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _compact, _labels, centers = cv2.kmeans(samples, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # фон — кластер с меньшим L* (темнее)
    bg_idx = int(np.argmin(centers[:, 0]))

    # ---- 3) строим маску ФОНА для всего кадра
    L = lab[:, :, 0]; A = lab[:, :, 1]; B = lab[:, :, 2]
    c_bg = centers[bg_idx]
    c_oth = centers[1 - bg_idx]
    # метрика — квадрат евклидовой дистанции в LAB
    d_bg  = (L - c_bg[0])**2 + (A - c_bg[1])**2 + (B - c_bg[2])**2
    d_oth = (L - c_oth[0])**2 + (A - c_oth[1])**2 + (B - c_oth[2])**2
    bg_mask = (d_bg <= d_oth).astype(np.uint8) * 255

    # немного почистим маску фона
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 1)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), 1)
    bg_mask = cv2.medianBlur(bg_mask, 5)

    # если фон «захватил всё» — лучше не трогать
    if bg_mask.mean() > 225:
        return bgr

    # ---- 4) ищем границы там, где фон уходит
    row_bg = (bg_mask.mean(axis=1) / 255.0).astype(np.float32)
    col_bg = (bg_mask.mean(axis=0) / 255.0).astype(np.float32)

    # сгладим проекции
    k = 9
    row_bg = cv2.GaussianBlur(row_bg.reshape(-1,1), (1, k), 0).ravel()
    col_bg = cv2.GaussianBlur(col_bg.reshape(1,-1), (k, 1), 0).ravel()

    thr = 0.22  # если доля фона ниже этого, считаем «бумага началась»
    sust_rows = max(6, int(round(0.015 * h)))
    sust_cols = max(6, int(round(0.015 * w)))

    # top
    y0 = 0; run = 0
    for y in range(h):
        if row_bg[y] < thr:
            run += 1
            if run >= sust_rows:
                y0 = max(0, y - sust_rows + 1)
                break
        else:
            run = 0
    # bottom
    y1 = h - 1; run = 0
    for y in range(h-1, -1, -1):
        if row_bg[y] < thr:
            run += 1
            if run >= sust_rows:
                y1 = min(h-1, y + sust_rows - 1)
                break
        else:
            run = 0
    # left
    x0 = 0; run = 0
    for x in range(w):
        if col_bg[x] < thr:
            run += 1
            if run >= sust_cols:
                x0 = max(0, x - sust_cols + 1)
                break
        else:
            run = 0
    # right
    x1 = w - 1; run = 0
    for x in range(w-1, -1, -1):
        if col_bg[x] < thr:
            run += 1
            if run >= sust_cols:
                x1 = min(w-1, x + sust_cols - 1)
                break
        else:
            run = 0

    # добавим/учтём margin и небольшой «наружный» пад
    m = int(round(margin_px * scale))
    x0 = max(0, x0 + m); y0 = max(0, y0 + m)
    x1 = min(w-1, x1 - m); y1 = min(h-1, y1 - m)

    pad = max(2, int(round(0.005 * min(h, w))))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w-1, x1 + pad); y1 = min(h-1, y1 + pad)

    # sanity
    if x1 - x0 < 30 or y1 - y0 < 30:
        return bgr

    # ---- 5) маппинг в исходный размер + предохранители
    X0 = int(round(x0 / scale)); Y0 = int(round(y0 / scale))
    X1 = int(round(x1 / scale)); Y1 = int(round(y1 / scale))
    newW = X1 - X0 + 1; newH = Y1 - Y0 + 1
    if newW < 50 or newH < 50:
        return bgr
    if (W - newW)/W > max_shrink_ratio or (H - newH)/H > max_shrink_ratio:
        return bgr

    return bgr[Y0:Y1+1, X0:X1+1].copy()

# ---------- ГЕОМЕТРИЯ ----------

def _predicted_size_from_pts(pts):
    r = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = r
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    return max(widthA, widthB), max(heightA, heightB)

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
    return - (a1+a2+a3+a4)

def _is_valid_quad(approx, img_area, min_area_ratio=0.10, max_area_ratio=0.98,
                   min_aspect=0.3, max_aspect=3.8, min_side_px=160):
    if len(approx) != 4: return False
    area = abs(cv2.contourArea(approx))
    if area <= 0: return False
    area_ratio = area / float(img_area)
    if not (min_area_ratio <= area_ratio <= max_area_ratio): return False
    if not cv2.isContourConvex(approx): return False
    w_pred, h_pred = _predicted_size_from_pts(approx.reshape(4,2))
    if w_pred < min_side_px or h_pred < min_side_px: return False
    asp = w_pred / max(h_pred, 1e-6)
    if not (min_aspect <= asp <= max_aspect): return False
    return True

def _deskew_angle(edges):
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is None: return 0.0
    angles = []
    for rho, theta in lines[:,0,:]:
        ang = (theta * 180/np.pi) % 180
        if ang > 90: ang -= 180
        if abs(ang) < 30 or abs(abs(ang)-90) < 30:
            angles.append(ang)
    if not angles: return 0.0
    angles = np.array(angles)
    m = np.median(angles)
    good = angles[np.abs(angles - m) < 15]
    return float(np.mean(good)) if len(good) else float(m)

# ---------- ПОИСК КВАДРАТА (с «угнетением» фона стола) ----------

def _find_best_quad(small_bgr):
    h, w = small_bgr.shape[:2]
    img_area = h*w

    # чуть подавим стол: уменьшим контрасты не-белых областей,
    # чтобы границы у светлой бумаги стали доминировать
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    paper_like = (s < 70) & (v > 160)
    atten = cv2.blur((~paper_like).astype(np.uint8) * 255, (9,9))
    atten = atten.astype(np.float32) / 255.0  # 0..1
    # ослабим яркостные перепады фона
    small_bgr_att = cv2.addWeighted(small_bgr, 1.0, small_bgr, 0.0, 0)
    small_bgr_att = (small_bgr_att.astype(np.float32) * (1.0 - 0.25*atten[...,None])).astype(np.uint8)

    gray = cv2.cvtColor(small_bgr_att, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = _auto_canny(gray_blur, sigma=0.33)

    # HED как усилитель
    hed = HED(Path(__file__).resolve().parent.parent / "models" / "hed")
    if hed.available:
        hed_map = hed.predict(small_bgr_att)
        if hed_map is not None:
            hed_bin = cv2.threshold(hed_map, 40, 255, cv2.THRESH_BINARY)[1]
            hed_bin = cv2.GaussianBlur(hed_bin, (3,3), 0)
            mix = cv2.addWeighted(canny, 0.7, hed_bin, 0.9, 0)
            _, edges = cv2.threshold(mix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            edges = canny
    else:
        edges = canny

    # морфология и deskew
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)),1)

    ang = _deskew_angle(edges)
    if abs(ang) > 0.8:
        small_bgr_att = rotate_bound(small_bgr_att, -ang)
        gray = cv2.cvtColor(small_bgr_att, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny = _auto_canny(gray_blur, sigma=0.33)
        if hed.available:
            hed_map = hed.predict(small_bgr_att)
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
            if score > best_score: best, best_score = approx, score
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32).reshape(-1,1,2)
        if _is_valid_quad(box, img_area):
            score = _rect_score(box.reshape(4,2)) + 0.00015 * cv2.contourArea(c)
            if score > best_score: best, best_score = box, score

    return best, small_bgr_att

# ---------- ОСНОВНОЙ ПАЙПЛАЙН ----------

def scan(
    image_bgr: np.ndarray,
    output_mode: str = "bw",
    debug: bool = False,
    target_long_side: int | None = 2400,
    trim: bool = True,
    trim_margin_px: int = 8,
    **kwargs,
) -> np.ndarray:
    # нормализация цвета/шума
    img = _gray_world_wb(image_bgr)
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

    # нормализуем размер
    H, W = img.shape[:2]
    scale = 1300.0 / max(H, W)
    small = cv2.resize(img, (int(W*scale), int(H*scale))) if scale < 1.0 else img.copy()
    ratio = image_bgr.shape[0] / float(small.shape[0])

    quad, _small_used = _find_best_quad(small)
    if quad is not None:
        pts = quad.reshape(4,2) * ratio
        doc = four_point_transform(
            image_bgr, pts,
            min_side=260,
            target_long_side=target_long_side,
            inter=cv2.INTER_CUBIC
        )
    else:
        doc = image_bgr.copy()

    # === НОВОЕ: авто-трим по маске бумаги ===
    if trim:
        doc = _auto_trim_to_mask(doc, margin_px=trim_margin_px, min_keep_ratio=0.55)

    # улучшение читаемости
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
        return out, {"small": _small_used}
    return out
