from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

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
                self.net = cv2.dnn.readNetFromCaffe(str(self.proto), str(self.weights))
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
            blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0/255.0,
                                         size=(W, H), mean=(0,0,0),
                                         swapRB=False, crop=False)
            self.net.setInput(blob)
            ed = self.net.forward()  # (1,1,H,W)
            ed = ed[0, 0]
            ed = (ed * 255.0).clip(0, 255).astype(np.uint8)
            return ed
        except Exception as e:
            print(f"[HED] forward error: {e}")
            return None
