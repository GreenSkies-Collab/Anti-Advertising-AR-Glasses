"""
Ad-Blocking Glasses — Prototype v0.1
=====================================
Proof-of-concept: live webcam feed with real-time detection and blur overlay.

This first version uses YOLOv8-nano (pre-trained on COCO) to detect
screen-like objects (tv, laptop, cell phone) as a proxy for ad surfaces.
Replace TARGET_CLASSES and swap in a custom-trained model to target real ads.

Controls:
  Q       — quit
  B       — cycle blur style (gaussian / pixelate / black box)
  +/-     — increase / decrease blur strength
  D       — toggle detection boxes (debug mode)
  SPACE   — pause / resume
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "yolov8n.pt"          # downloaded automatically on first run
CAMERA_INDEX = 0                    # 0 = default webcam
CONFIDENCE_THRESHOLD = 0.45        # detections below this are ignored
TARGET_CLASSES = {                  # COCO class ids to treat as "ad surfaces"
    62: "tv",
    63: "laptop",
    67: "cell phone",
}
BLUR_STYLES = ["gaussian", "pixelate", "blackbox"]
INITIAL_BLUR_STRENGTH = 25          # must be odd for gaussian; controls block size for pixelate
WINDOW_NAME = "Ad-Blocking Glasses — Prototype"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
               style: str, strength: int) -> np.ndarray:
    """Apply the selected blur style to the bounding-box region."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    if style == "gaussian":
        k = strength if strength % 2 == 1 else strength + 1  # kernel must be odd
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
    elif style == "pixelate":
        block = max(1, strength // 5)
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                           interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:  # blackbox
        blurred = np.zeros_like(roi)

    frame[y1:y2, x1:x2] = blurred
    return frame


def draw_hud(frame: np.ndarray, style: str, strength: int,
             debug: bool, paused: bool, fps: float) -> None:
    """Render a minimal HUD in the top-left corner."""
    h, w = frame.shape[:2]
    lines = [
        f"Mode: {style}  |  Strength: {strength}",
        f"FPS: {fps:.1f}  |  Debug: {'on' if debug else 'off'}",
        "Q=quit  B=style  +/-=strength  D=debug  SPACE=pause",
    ]
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("Loading model…")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    blur_style_idx = 0
    blur_strength = INITIAL_BLUR_STRENGTH
    debug = False
    paused = False
    fps = 0.0
    prev_tick = cv2.getTickCount()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("Running — press Q to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        # --- Controls ---
        if key == ord("q"):
            break
        elif key == ord("b"):
            blur_style_idx = (blur_style_idx + 1) % len(BLUR_STYLES)
        elif key == ord("+") or key == ord("="):
            blur_strength = min(101, blur_strength + 2)
        elif key == ord("-"):
            blur_strength = max(3, blur_strength - 2)
        elif key == ord("d"):
            debug = not debug
        elif key == ord(" "):
            paused = not paused

        if paused:
            # Still render the frame so HUD updates
            cv2.imshow(WINDOW_NAME, frame if "frame" in dir() else np.zeros((720, 1280, 3), dtype=np.uint8))
            continue

        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — retrying…")
            continue

        # --- Inference ---
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        boxes = results[0].boxes

        # --- Render blur ---
        current_style = BLUR_STYLES[blur_style_idx]
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Clamp to frame boundaries
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            frame = apply_blur(frame, x1, y1, x2, y2, current_style, blur_strength)

            if debug:
                conf = float(box.conf[0])
                label = f"{TARGET_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        # --- FPS ---
        tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (tick - prev_tick)
        prev_tick = tick

        draw_hud(frame, current_style, blur_strength, debug, paused, fps)
        cv2.imshow(WINDOW_NAME, frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
