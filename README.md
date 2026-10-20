# Ad-Blocking Glasses 🕶️🚫📺

**An ad blocker for the real world.** A proof-of-concept that watches a live camera feed, detects ad-like surfaces in real time, and blurs them out before you see them. It's the software half of a pair of AR glasses that filter advertising from your field of view.

> Status: **prototype v0.1**. It runs on a laptop webcam today. The long-term goal is to run it on a head-mounted camera and display.

---

## How it works

```
 ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────┐
 │  Camera  │ ─► │  YOLOv8-nano     │ ─► │  Filter by class │ ─► │  Blur /  │ ─► Display
 │  frame   │    │  object detector │    │  ("ad surfaces") │    │  mask    │
 └──────────┘    └──────────────────┘    └──────────────────┘    └──────────┘
```

1. **Capture**: grab a frame from the webcam (1280×720 when the camera supports it).
2. **Detect**: run [YOLOv8-nano](https://docs.ultralytics.com/models/yolov8/), a small, fast object detector pre-trained on the COCO dataset.
3. **Filter**: keep only detections of classes treated as ad surfaces. COCO has no "advertisement" class, so v0.1 uses screens as a stand-in: `tv`, `laptop` and `cell phone`.
4. **Mask**: apply a Gaussian blur, pixelation, or a solid black box to each detected region.
5. **Display**: show the filtered frame with a small HUD (mode, strength, FPS).

## Quick start

Requires **Python 3.9+** and a webcam.

```bash
git clone https://github.com/GreenSkies-Collab/Anti-Advertising-AR-Glasses.git
cd Anti-Advertising-AR-Glasses

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

On the first run, Ultralytics downloads the `yolov8n.pt` weights (~6 MB) automatically.

## Controls

| Key       | Action                                              |
|-----------|-----------------------------------------------------|
| `Q`       | Quit                                                |
| `B`       | Cycle blur style: gaussian → pixelate → black box   |
| `+` / `-` | Increase / decrease blur strength                   |
| `D`       | Toggle debug mode (boxes, labels, confidence)       |
| `Space`   | Pause / resume                                      |

## Configuration

All settings are constants at the top of [`main.py`](main.py):

| Setting                 | Default        | What it does                                         |
|-------------------------|----------------|------------------------------------------------------|
| `MODEL_PATH`            | `yolov8n.pt`   | Detection model weights (swap in a custom model)     |
| `CAMERA_INDEX`          | `0`            | Which camera to use                                  |
| `CONFIDENCE_THRESHOLD`  | `0.45`         | Ignore detections below this confidence              |
| `TARGET_CLASSES`        | tv, laptop, phone | COCO class IDs to treat as ad surfaces            |
| `INITIAL_BLUR_STRENGTH` | `25`           | Starting blur kernel / pixel-block size              |

The full list of COCO class IDs is in the [Ultralytics docs](https://docs.ultralytics.com/datasets/detect/coco/).

## Roadmap

- [x] **v0.1**: real-time detect-and-blur pipeline on a webcam, using screens as a stand-in for ads
- [ ] **Custom ad detector**: fine-tune YOLOv8 on a labelled dataset of billboards, posters, bus-stop ads, shop signage and logos
- [ ] **Temporal smoothing**: track boxes across frames so the blur doesn't flicker when a detection drops out for a frame
- [ ] **Replace instead of blur**: paint over ads with something useful or calm (art, weather, a plain wall texture)
- [ ] **Edge hardware**: run on a Raspberry Pi 5 / Jetson Orin Nano with a head-mounted camera
- [ ] **Real AR output**: drive a see-through display (e.g. passthrough or optical AR glasses) instead of a desktop window

## Limitations

- **It doesn't detect ads yet.** v0.1 blurs *every* TV, laptop and phone it sees, ads or not. Real ad detection needs a custom-trained model.
- **Latency matters on glasses.** A desktop GPU easily reaches real-time; a CPU-only laptop gets fewer FPS. Wearable hardware will need a smaller or quantised model.
- **Privacy.** The prototype processes everything locally and saves nothing, which is how it should stay.

## Project structure

```
.
├── main.py            # The full prototype: capture → detect → blur → display
├── requirements.txt   # Python dependencies
├── LICENSE
└── README.md
```

## Built with

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): object detection
- [OpenCV](https://opencv.org/): camera capture, image processing and display
- [NumPy](https://numpy.org/)

> **Licence note:** Ultralytics YOLOv8 is licensed under **AGPL-3.0**. If you distribute this project or offer it as a service, check the AGPL terms or get an Ultralytics Enterprise licence.

## Contributing

Ideas, issues and pull requests are welcome, especially datasets or labelled images of real-world advertising.

## License

[MIT](LICENSE). Note that the dependencies carry their own licences (see above).
