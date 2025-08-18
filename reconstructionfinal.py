import os
import time
from typing import Tuple
import cv2
import numpy as np


# CONFIG: set your input image path here
   
IMAGE_PATH = "Moon_image_dataset/north pole/north pole_1.png"
MAX_WIDTH = 900
APPROX_EPS = 0.01   # polygon simplification ratio
THICKNESS = 2
SMOOTH = True


   
# Utility functions
   

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def resize_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def binarize(mask_gray: np.ndarray, use_otsu: bool = True) -> np.ndarray:
    if use_otsu:
        _ , th = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _ , th = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
    return th

def morphological_cleanup(binary: np.ndarray, k_open: int = 3, k_close: int = 3) -> np.ndarray:
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_o, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_c, iterations=1)
    return closed

def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-10:
        return 99.0
    PIX_MAX = 255.0
    return 20.0 * np.log10(PIX_MAX) - 10.0 * np.log10(m)

def iou(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    a = (a_bin > 0).astype(np.uint8)
    b = (b_bin > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)

   
# Drawing Canvas with OpenCV
   
class Drawer:
    def __init__(self, base_bgr: np.ndarray):
        self.base = base_bgr.copy()
        self.canvas = base_bgr.copy()
        self.overlay = np.zeros_like(base_bgr)  # Stores strokes only
        self.drawing = False
        self.brush = 6
        self.eraser = False
        self.last_pt = None
        self.win = "Draw on Image (q: finish, s: save, e: eraser, +/-: brush, c: clear, r: reset)"

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_pt = (x, y)
            self._stroke(self.last_pt, (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._stroke(self.last_pt, (x, y))
            self.last_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self._stroke(self.last_pt, (x, y))
            self.last_pt = None

    def _stroke(self, p0, p1):
        if p0 is None or p1 is None:
            return
        color = (0, 0, 0) if self.eraser else (255, 255, 255)
        thickness = max(1, self.brush)
        cv2.line(self.overlay, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        self.canvas = cv2.addWeighted(self.base, 1.0, self.overlay, 1.0, 0)

    def loop(self) -> np.ndarray:
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        snapshot_idx = 0
        while True:
            disp = self.canvas.copy()
            hud = f"brush:{self.brush} | eraser:{'ON' if self.eraser else 'off'} | keys: +/- e c r s q"
            cv2.putText(disp, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.win, disp)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                self.brush = min(128, self.brush + 1)
            elif key == ord('-'):
                self.brush = max(1, self.brush - 1)
            elif key == ord('e'):
                self.eraser = not self.eraser
            elif key == ord('c') or key == ord('r'):
                self.overlay[:] = 0
                self.canvas = self.base.copy()
            elif key == ord('s'):
                cv2.imwrite(f"draw_snapshot_{snapshot_idx}.png", self.canvas)
                snapshot_idx += 1
        cv2.destroyWindow(self.win)
        return self.canvas

   
# Pattern Extraction & Regeneration
   
def extract_pattern(base_bgr: np.ndarray, drawn_bgr: np.ndarray) -> np.ndarray:
    base_gray = to_gray(base_bgr)
    drawn_gray = to_gray(drawn_bgr)
    diff = cv2.absdiff(drawn_gray, base_gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    th = binarize(diff, use_otsu=True)
    th = morphological_cleanup(th, 3, 5)
    return th

def regenerate_from_contours(pattern_bin: np.ndarray, approx_eps_ratio: float = 0.01,
                              thickness: int = 2, smooth: bool = True) -> np.ndarray:
    contours, _ = cv2.findContours(pattern_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = pattern_bin.shape[:2]
    # out = np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w, 3), dtype=np.uint8)  # 3-channel image for color overlay
    base_colored = cv2.cvtColor(pattern_bin, cv2.COLOR_GRAY2BGR)
    out[:] = base_colored  # initialize with gray pattern image
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, approx_eps_ratio * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if smooth and len(approx) >= 3:
            pts = approx.reshape(-1, 2).astype(np.float32)
            for _ in range(2):
                new_pts = []
                for i in range(len(pts)):
                    p = pts[i]
                    q = pts[(i + 1) % len(pts)]
                    new_pts.append(0.75 * p + 0.25 * q)
                    new_pts.append(0.25 * p + 0.75 * q)
                pts = np.array(new_pts, dtype=np.float32)
            approx = pts.reshape(-1, 1, 2).astype(np.int32)
        # cv2.polylines(out, [approx], isClosed=True, color=255, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.polylines(out, [approx], isClosed=True, color=(0,255,0), thickness=thickness, lineType=cv2.LINE_AA)
    return out

def visualize_and_save(original_bin: np.ndarray, regenerated_bin: np.ndarray, out_dir: str) -> None:
    ensure_dir(out_dir)

    # Ensure same size
    if original_bin.shape != regenerated_bin.shape[:2]:
        regenerated_bin = cv2.resize(regenerated_bin, (original_bin.shape[1], original_bin.shape[0]))

    # Convert regenerated to grayscale if it's 3-channel
    if len(regenerated_bin.shape) == 3 and regenerated_bin.shape[2] == 3:
        regenerated_gray = cv2.cvtColor(regenerated_bin, cv2.COLOR_BGR2GRAY)
    else:
        regenerated_gray = regenerated_bin

    # Now absdiff works
    abs_err = cv2.absdiff(original_bin, regenerated_gray)

    heat = cv2.applyColorMap(abs_err, cv2.COLORMAP_JET)
    m = mse(original_bin, regenerated_gray)
    p = psnr(original_bin, regenerated_gray)
    j = iou(original_bin, regenerated_gray)

    base_color = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    base_color = resize_to_max_width(base_color, MAX_WIDTH)

    # Overlay original pattern (blue)
    A = base_color.copy()
    mask = (original_bin > 0)
    A[mask] = (255, 0, 0)

    # Overlay regenerated pattern (green)
    B = base_color.copy()
    mask = (regenerated_gray > 0)
    B[mask] = (0, 255, 0)

    # Error heatmap overlay
    C = cv2.addWeighted(base_color, 0.7, heat, 0.6, 0)

    cv2.putText(A, "Original Pattern", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(B, "Regenerated", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(C, f"Error Heatmap | MSE:{m:.2f} PSNR:{p:.2f} IoU:{j:.3f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.hstack([A, B, C])
    cv2.imshow("Original | Regenerated | Error", combined)
    cv2.waitKey(0)
    cv2.destroyWindow("Original | Regenerated | Error")

    cv2.imwrite(os.path.join(out_dir, "pattern_original.png"), original_bin)
    cv2.imwrite(os.path.join(out_dir, "pattern_regenerated.png"), regenerated_gray)
    cv2.imwrite(os.path.join(out_dir, "error_heatmap.png"), heat)
    cv2.imwrite(os.path.join(out_dir, "comparison.png"), combined)

   
# Main
   
def main():
    base = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    base = resize_to_max_width(base, MAX_WIDTH)
    drawer = Drawer(base)
    drawn = drawer.loop()
    pattern_bin = extract_pattern(base, drawn)
    regenerated = regenerate_from_contours(pattern_bin, approx_eps_ratio=APPROX_EPS,
                                          thickness=THICKNESS, smooth=SMOOTH)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", stamp)
    visualize_and_save(pattern_bin, regenerated, out_dir)
    print("Saved outputs to:", out_dir)

if __name__ == "__main__":
    main()