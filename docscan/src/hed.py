from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


class _CenterCrop:
    """Crop HED feature maps around their centre, preserving pixel alignment."""

    def __init__(self, params, blobs):
        pass

    def getMemoryShapes(self, inputs):
        source, target = inputs
        self.y = (source[2] - target[2]) // 2
        self.x = (source[3] - target[3]) // 2
        self.height, self.width = target[2:4]
        return [[source[0], source[1], self.height, self.width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y:self.y + self.height, self.x:self.x + self.width]]


cv2.dnn_registerLayer("DocScanCenterCrop", _CenterCrop)


class HED:
    """
    Лёгкая обёртка над OpenCV DNN HED.
    Ищет модели в: <repo_root>/docscan/models/hed/{deploy.prototxt, hed_pretrained_bsds.caffemodel}
    Если нет моделей — available=False и predict() вернёт None.
    """
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.proto = self.models_dir / "deploy.prototxt"
        self.weights = self.models_dir / "hed_pretrained_bsds.caffemodel"
        self.net = None
        self.available = self.proto.is_file() and self.weights.is_file()
        if self.available:
            try:
                # Use a private layer name rather than overriding OpenCV's Crop
                # globally. Default Caffe Crop uses a zero offset and shifts edges.
                config = self.proto.read_text(encoding="utf-8").replace('type: "Crop"', 'type: "DocScanCenterCrop"')
                self.net = cv2.dnn.readNetFromCaffe(
                    np.frombuffer(config.encode("utf-8"), dtype=np.uint8),
                    np.fromfile(self.weights, dtype=np.uint8),
                )
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print(f"[HED] load error: {e}")
                self.available = False

    def predict(self, img_small) -> np.ndarray | None:
        """
        img_small: uint8 HxW (GRAY) или HxWx3 (BGR)
        return: uint8 HxW (0..255) edge map или None
        """
        if not self.available or self.net is None:
            return None
        try:
            if img_small.ndim == 2:
                bgr = cv2.cvtColor(img_small, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img_small
            H, W = bgr.shape[:2]
            # The original BSDS HED model expects mean-subtracted BGR values
            # in the 0..255 range, not a normalised 0..1 image.
            blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0,
                                         size=(W, H), mean=(104.00698793, 116.66876762, 122.67891434),
                                         swapRB=False, crop=False)
            self.net.setInput(blob)
            ed = self.net.forward()  # (1,1,H,W)
            ed = ed[0, 0]
            if ed.shape != (H, W):
                ed = cv2.resize(ed, (W, H), interpolation=cv2.INTER_LINEAR)
            ed = (ed * 255.0).clip(0, 255).astype(np.uint8)
            return ed
        except Exception as e:
            print(f"[HED] forward error: {e}")
            return None
