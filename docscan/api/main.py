from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from typing import List, Literal
import numpy as np
import cv2
from pathlib import Path
from urllib.parse import quote
import io
import zipfile
import img2pdf
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docscan.src import scan
from docscan.src.editing import apply_edit, rotation_layout
from docscan.src.geometry import order_points
from docscan.src.pipeline import suggest_crop

app = FastAPI(title="DocScan API", version="0.7.0")

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_BATCH_FILES = 20

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
WEB_DIR = BASE_DIR / "web"
INDEX_HTML = WEB_DIR / "index.html"

def _np_to_bytes(img, fmt: Literal["png", "jpg"] = "png", quality=92) -> bytes:
    if img.ndim == 2:
        ok, buf = cv2.imencode(f".{fmt}", img, [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt=="jpg" else [])
    else:
        if fmt == "jpg":
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Encode failed")
    return buf.tobytes()

def _file_stem(filename: str | None) -> str:
    name = (filename or "scan").replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(name).stem
    stem = "".join(c for c in stem if c.isprintable() and c not in '<>:"|?*')
    return stem.strip(" .")[:100] or "scan"


def _disposition(filename: str, kind: str = "attachment") -> str:
    # RFC 5987 supports Cyrillic names without putting Unicode into HTTP headers.
    suffix = Path(filename).suffix
    return f"{kind}; filename=\"scan{suffix}\"; filename*=UTF-8''{quote(filename, safe='')}"


def _decode_image(data: bytes, filename: str | None) -> np.ndarray:
    label = filename or "image"
    if not data:
        raise HTTPException(400, f"Файл «{label}» пуст")
    try:
        # Check dimensions before decoding; normalise phone orientation and
        # flatten transparent editor masks onto white paper.
        with Image.open(io.BytesIO(data)) as header:
            if header.width * header.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, f"Изображение «{label}» превышает 40 мегапикселей")
            image = ImageOps.exif_transpose(header)
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                image = Image.new("RGB", rgba.size, "white")
                image.paste(rgba, mask=rgba.getchannel("A"))
            rgb = np.array(image.convert("RGB"))
    except Image.DecompressionBombError:
        raise HTTPException(413, f"Изображение «{label}» слишком большое")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, f"Не удалось прочитать изображение «{label}»")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


async def _read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"Файл «{upload.filename or 'image'}» превышает 20 МБ")
    return await run_in_threadpool(_decode_image, data, upload.filename)


def _make_pdf(pages: list[bytes]) -> bytes:
    return img2pdf.convert(pages, layout_fun=img2pdf.get_fixed_dpi_layout_fun((300, 300)))

@app.get("/")
def root():
    return FileResponse(INDEX_HTML)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class EditSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    rotation: float = Field(0, ge=-360, le=360)
    crop: tuple[float, float, float, float] | None = None
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None = None


def _decode_mask(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as mask:
            if mask.format != "PNG" or mask.size != (shape[1], shape[0]) or "A" not in mask.getbands():
                raise ValueError("Нужна прозрачная PNG-маска того же размера, что и изображение")
            return np.array(mask.getchannel("A"))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"Не удалось прочитать маску: {exc}")


async def _edited_pixels(file: UploadFile, settings: str, mask: UploadFile | None):
    try:
        edit = EditSettings.model_validate_json(settings)
    except ValidationError:
        raise HTTPException(422, "Некорректные настройки редактора")
    image = await _read_image(file)
    alpha = None
    if mask is not None:
        data = await mask.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, "Маска превышает 20 МБ")
        alpha = await run_in_threadpool(_decode_mask, data, image.shape[:2])
    try:
        result = await run_in_threadpool(apply_edit, image, **edit.model_dump(), mask=alpha, max_pixels=MAX_IMAGE_PIXELS)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return result


@app.post("/edit")
async def edit_single(
    file: UploadFile = File(...),
    settings: str = Form(...),
    mask: UploadFile | None = File(None),
):
    result = await _edited_pixels(file, settings, mask)
    encoded = await run_in_threadpool(_np_to_bytes, result, "png")
    fmt = "png"
    # Large PNGs must remain uploadable to /scan or to another editing pass.
    if len(encoded) > MAX_FILE_BYTES:
        encoded = await run_in_threadpool(_np_to_bytes, result, "jpg", 95)
        fmt = "jpg"
    if len(encoded) > MAX_FILE_BYTES:
        raise HTTPException(413, "Результат превышает 20 МБ; уменьшите область выделения")
    return Response(encoded, media_type="image/png" if fmt == "png" else "image/jpeg",
                    headers={"Content-Disposition": _disposition(f"{_file_stem(file.filename)}.{fmt}")})


@app.post("/preview")
async def preview_edit(
    file: UploadFile = File(...),
    settings: str = Form("{}"),
    mask: UploadFile | None = File(None),
    mode: Literal["bw", "gray", "color"] = Query("bw"),
    long_side: int = Query(2400, ge=600, le=6000),
    text_strength: float = Query(0.5, ge=0, le=1),
):
    image = await _edited_pixels(file, settings, mask)
    result = await run_in_threadpool(scan, image, output_mode=mode, target_long_side=long_side,
                                     align=False, trim=False, text_strength=text_strength)
    encoded = await run_in_threadpool(_np_to_bytes, result, "png")
    return Response(encoded, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/detect")
async def detect_crop(
    file: UploadFile = File(...),
    rotation: float = Query(0, ge=-360, le=360),
    align: bool = Query(True),
    trim: bool = Query(True),
    trim_margin: int = Query(0, ge=0, le=40),
    trim_thr: float = Query(0.22, ge=0.05, le=0.6),
    trim_sustain: int = Query(6, ge=3, le=50),
    trim_pad: int = Query(12, ge=0, le=40),
):
    if not np.isfinite(rotation):
        raise HTTPException(422, "Некорректный угол поворота")
    image = await _read_image(file)
    if min(image.shape[:2]) < 2:
        raise HTTPException(422, "Изображение слишком маленькое для поиска границ")
    try:
        result = await run_in_threadpool(suggest_crop, image, align=align, trim=trim, trim_margin_px=trim_margin,
                                         _trim_thr=trim_thr, _trim_sustain=trim_sustain, _trim_pad=trim_pad)
    except ValueError:
        raise HTTPException(422, "Не удалось найти границы; выделите область вручную")
    matrix, width, height = rotation_layout(image.shape[1], image.shape[0], rotation)
    points = cv2.perspectiveTransform(np.array([result["quad"]], np.float64), matrix)[0]
    points = order_points(np.clip(points, [0, 0], [width, height]).astype(np.float32))
    x, y = points.min(axis=0)
    w, h = points.max(axis=0) - [x, y]
    return {**result, "quad": points.tolist(), "crop": [float(x), float(y), float(w), float(h)],
            "kind": "persp" if result["found"] else "rect"}


@app.post("/scan")
async def scan_single(
    file: UploadFile = File(...),
    mode: Literal["bw","gray","color"] = Query("bw"),
    long_side: int | None = Query(2400, ge=600, le=6000),
    fmt: Literal["png","jpg","pdf"] = Query("png"),
    text_strength: float = Query(0.5, ge=0, le=1),
    # выравнивание и трим
    align: bool = Query(True),
    trim: bool = Query(True),
    trim_margin: int = Query(0, ge=0, le=40),
    # ручки трима:
    trim_thr: float = Query(0.22, ge=0.05, le=0.6),
    trim_sustain: int = Query(6, ge=3, le=50),
    trim_pad: int = Query(12, ge=0, le=40),
):
    img_bgr = await _read_image(file)
    result = await run_in_threadpool(
        scan,
        img_bgr,
        output_mode=mode,
        target_long_side=long_side,
        align=align,
        trim=trim,
        trim_margin_px=trim_margin,
        _trim_thr=trim_thr,
        _trim_sustain=trim_sustain,
        _trim_pad=trim_pad,
        text_strength=text_strength,
    )

    if fmt == "pdf":
        page = await run_in_threadpool(_np_to_bytes, result, "png" if result.ndim == 2 else "jpg")
        pdf_bytes = await run_in_threadpool(_make_pdf, [page])
        return Response(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": _disposition(f"{_file_stem(file.filename)}.pdf")},
        )
    else:
        b = await run_in_threadpool(_np_to_bytes, result, fmt=fmt)
        mt = "image/png" if fmt=="png" else "image/jpeg"
        return Response(
            b,
            media_type=mt,
            headers={"Content-Disposition": _disposition(f"{_file_stem(file.filename)}.{fmt}", "inline")}
        )

@app.post("/scan/batch")
async def scan_batch(
    files: List[UploadFile] = File(...),
    mode: Literal["bw","gray","color"] = Query("bw"),
    long_side: int | None = Query(2400, ge=600, le=6000),
    out: Literal["zip","pdf"] = Query("pdf"),
    img_fmt: Literal["png","jpg"] = Query("jpg"),
    preserve: List[int] = Query(default=[]),
    text_strength: float = Query(0.5, ge=0, le=1),
    # выравнивание и трим
    align: bool = Query(True),
    trim: bool = Query(True),
    trim_margin: int = Query(0, ge=0, le=40),
    # ручки трима:
    trim_thr: float = Query(0.22, ge=0.05, le=0.6),
    trim_sustain: int = Query(6, ge=3, le=50),
    trim_pad: int = Query(12, ge=0, le=40),
):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"За один раз можно обработать не более {MAX_BATCH_FILES} файлов")
    if any(index < 0 or index >= len(files) for index in preserve):
        raise HTTPException(422, "Некорректный номер страницы с ручными правками")
    results: list[tuple[str, bytes]] = []
    used_names: set[str] = set()
    for index, uf in enumerate(files):
        img_bgr = await _read_image(uf)
        res = await run_in_threadpool(
            scan,
            img_bgr,
            output_mode=mode,
            target_long_side=long_side,
            align=align and index not in preserve,
            trim=trim and index not in preserve,
            trim_margin_px=trim_margin,
            _trim_thr=trim_thr,
            _trim_sustain=trim_sustain,
            _trim_pad=trim_pad,
            text_strength=text_strength,
        )
        stem = _file_stem(uf.filename)
        unique_stem = stem
        counter = 2
        while unique_stem.casefold() in used_names:
            unique_stem = f"{stem}-{counter}"
            counter += 1
        used_names.add(unique_stem.casefold())
        encoding = img_fmt if out == "zip" else ("png" if res.ndim == 2 else "jpg")
        encoded = await run_in_threadpool(_np_to_bytes, res, encoding)
        results.append((unique_stem, encoded))

    if out == "pdf":
        pdf_bytes = await run_in_threadpool(_make_pdf, [data for _, data in results])
        return Response(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="scans.pdf"'}
        )
    else:
        with io.BytesIO() as bio:
            with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
                for stem, data in results:
                    zf.writestr(f"{stem}.{img_fmt}", data)
            return Response(
                bio.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="scans.zip"'}
            )
