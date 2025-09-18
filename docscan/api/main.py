from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal
import numpy as np
import cv2
from pathlib import Path
import io
import zipfile
import img2pdf
from PIL import Image

from docscan.src import scan

app = FastAPI(title="DocScan API", version="0.4.0")

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

def _np_to_pil(img) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    else:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

@app.get("/")
def root():
    return FileResponse(INDEX_HTML)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.post("/scan")
async def scan_single(
    file: UploadFile = File(...),
    mode: Literal["bw","gray","color"] = Query("bw"),
    long_side: int | None = Query(2400, ge=600, le=6000),
    fmt: Literal["png","jpg","pdf"] = Query("png"),
    # выравнивание и трим
    align: bool = Query(True),
    trim: bool = Query(True),
    trim_margin: int = Query(8, ge=0, le=40),
    # ручки трима:
    trim_thr: float = Query(0.22, ge=0.05, le=0.6),
    trim_sustain: int = Query(6, ge=3, le=50),
    trim_pad: int = Query(6, ge=0, le=40),
):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error":"Cannot decode image"}, status_code=400)

    result = scan(
        img_bgr,
        output_mode=mode,
        target_long_side=long_side,
        align=align,
        trim=trim,
        trim_margin_px=trim_margin,
        _trim_thr=trim_thr,
        _trim_sustain=trim_sustain,
        _trim_pad=trim_pad,
    )

    if fmt == "pdf":
        pil = _np_to_pil(result).convert("RGB")
        with io.BytesIO() as bio_img:
            pil.save(bio_img, format="JPEG", quality=92)
            pdf_bytes = img2pdf.convert([bio_img.getvalue()])
            return Response(
                pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{Path(file.filename).stem}.pdf"'}
            )
    else:
        b = _np_to_bytes(result, fmt=fmt)
        mt = "image/png" if fmt=="png" else "image/jpeg"
        return Response(
            b,
            media_type=mt,
            headers={"Content-Disposition": f'inline; filename="{Path(file.filename).stem}.{fmt}"'}
        )

@app.post("/scan/batch")
async def scan_batch(
    files: List[UploadFile] = File(...),
    mode: Literal["bw","gray","color"] = Query("bw"),
    long_side: int | None = Query(2400, ge=600, le=6000),
    out: Literal["zip","pdf"] = Query("pdf"),
    img_fmt: Literal["png","jpg"] = Query("jpg"),
    # выравнивание и трим
    align: bool = Query(True),
    trim: bool = Query(True),
    trim_margin: int = Query(8, ge=0, le=40),
    # ручки трима:
    trim_thr: float = Query(0.22, ge=0.05, le=0.6),
    trim_sustain: int = Query(6, ge=3, le=50),
    trim_pad: int = Query(6, ge=0, le=40),
):
    results: list[tuple[str, np.ndarray]] = []
    for uf in files:
        data = await uf.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        res = scan(
            img_bgr,
            output_mode=mode,
            target_long_side=long_side,
            align=align,
            trim=trim,
            trim_margin_px=trim_margin,
            _trim_thr=trim_thr,
            _trim_sustain=trim_sustain,
            _trim_pad=trim_pad,
        )
        results.append((Path(uf.filename).stem, res))

    if not results:
        return JSONResponse({"error":"No images processed"}, status_code=400)

    if out == "pdf":
        pages = [ _np_to_pil(r).convert("RGB") for _, r in results ]
        bio = io.BytesIO()
        pages[0].save(bio, format="PDF", save_all=True, append_images=pages[1:])
        return Response(
            bio.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="scans.pdf"'}
        )
    else:
        with io.BytesIO() as bio:
            with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
                for stem, img in results:
                    data = _np_to_bytes(img, fmt=img_fmt)
                    zf.writestr(f"{stem}.{img_fmt}", data)
            return Response(
                bio.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="scans.zip"'}
            )
