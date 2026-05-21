<h1 align="center">Gesture Vision</h1>

<p align="center">
  <strong>Control camera filters with your hands — no buttons, no clicks, just gestures.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MediaPipe-Hands%20%7C%20Face%20%7C%20Segmentation-brightgreen?logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/OpenCV-4.x-orange?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Web-GitHub%20Pages-purple?logo=github&logoColor=white" alt="GitHub Pages">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<p align="center">
  <em>Real-time hand-gesture-controlled camera filters using MediaPipe and OpenCV.<br>
  Available as a Python desktop app and a zero-install browser app.</em>
</p>

---

## Demo

> **Live demo:** [https://Armaan-Wadhwa.github.io/Gesture-Vision/](https://Armaan-Wadhwa.github.io/Gesture-Vision/)
>
> Replace `Armaan-Wadhwa` with your GitHub username after deploying.

<!-- Add a screenshot or GIF here -->
<!-- ![Demo](assets/demo.gif) -->

---

## Highlights

- **12 real-time filters** — Grayscale, Thermal, Invert, Sketch, Vintage, Cartoon, Pixelate, Blur, Edge, Sepia, HSV Shift, and more
- **7 hand gestures** — pinch, swipe, fist, open palm, thumbs up/down to control everything touchlessly
- **5 application modes** — apply filters to the full frame, background only, face only, a freehand-drawn region, or use a magnifier
- **Filter stacking** — chain multiple filters together (e.g. GRAY + INVERT + EDGE)
- **Adjustable intensity** — spread your hands apart to control filter strength in real time
- **Two platforms** — Python desktop app (OpenCV) and browser app (WebRTC + MediaPipe JS)
- **Zero server** — the web version runs 100% client-side; your camera feed never leaves your device

---

## Quick Start

### Option A: Browser (no install)

```bash
cd docs
python -m http.server 8000
# Open http://localhost:8000 in Chrome/Firefox/Edge
```

Or deploy to GitHub Pages (see [Deployment](#deploy-to-github-pages) below).

### Option B: Python Desktop

```bash
git clone https://github.com/Armaan-Wadhwa/Gesture-Vision.git
cd Gesture-Vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python hand-ditaction.py
```

Press **Q** to quit.

#### CLI Options

```bash
python hand-ditaction.py --camera 1              # use a different webcam
python hand-ditaction.py --width 640 --height 480  # lower resolution for speed
```

---

## Filters

| # | Filter | Effect |
|:-:|--------|--------|
| 0 | **None** | Original camera feed |
| 1 | **GRAY** | Grayscale |
| 2 | **THERMAL** | Jet colormap heat vision |
| 3 | **INVERT** | Negative / inverted colors |
| 4 | **SKETCH** | Pencil-drawing effect |
| 5 | **VINTAGE** | Warm sepia film tone |
| 6 | **CARTOON** | Bold edges + posterized color |
| 7 | **PIXELATE** | Mosaic / pixel art |
| 8 | **BLUR** | Gaussian bokeh blur |
| 9 | **EDGE** | Canny / Sobel edge detection |
| 10 | **SEPIA** | Sepia tone with vignette |
| 11 | **HSV_SHIFT** | Hue rotation (adjustable) |

---

## Modes

| Key | Mode | Description |
|:---:|------|-------------|
| `F` | **Normal** | Filter on full frame; when both hands visible, filter applies only inside the hand-box |
| `B` | **Background** | Filter on background only — you stay unfiltered (Selfie Segmentation) |
| `A` | **Face** | Filter on face region only (Face Mesh) |
| `D` | **Draw** | Trace a freehand shape with your index finger; filter fills the drawn area |
| `M` | **Magnifier** | Two-hand box zooms into the enclosed area |

---

## Gestures

| Gesture | Action |
|---------|--------|
| **Pinch** (right hand) | Next filter |
| **Pinch** (left hand) | Previous filter |
| **Open palm** (5 fingers) | Reset filter, clear stack, restore intensity |
| **Fist** (0 fingers) | Freeze / unfreeze frame |
| **Thumbs up** | Save snapshot |
| **Thumbs down** | Discard last snapshot |
| **Swipe** left / right | Switch filter |
| **Two-hand spread** | Thumb distance controls filter intensity (0%–100%) |

### Additional Controls (Desktop)

| Key | Action |
|:---:|--------|
| `N` | Toggle finger-count mode (show 1–4 fingers to jump to filter) |
| `0`–`9` | Toggle filter in/out of stack |
| `+` / `-` | Adjust HSV hue shift |
| `S` | Save snapshot |
| `C` | Clear drawn region |
| `Q` | Quit |

---

## Deploy to GitHub Pages

1. Push this repo to GitHub
2. Go to **Settings** > **Pages**
3. Source: **Deploy from a branch**
4. Branch: `main` — Folder: `/docs`
5. Save — site goes live in ~1 minute

The web version loads MediaPipe models from Google's CDN. Background segmentation and face mesh models load lazily (only when you switch to those modes).

---

## Project Structure

```
Gesture-Vision/
├── hand-ditaction.py    # Python desktop app (GestureVision class)
├── requirements.txt     # Python dependencies
├── TUTORIAL.md          # Detailed feature walkthrough
├── CLAUDE.md            # AI assistant context
├── docs/                # Web version (GitHub Pages)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── scratch/             # Experiments (not part of main app)
```

---

## Requirements

### Desktop

| Dependency | Version |
|------------|---------|
| Python | 3.8+ |
| opencv-python | 4.x |
| mediapipe | latest |
| numpy | latest |

### Web

- Modern browser (Chrome, Firefox, Edge, Safari 16+)
- Webcam access
- No install needed

---

## Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-filter`)
3. **Commit** your changes (`git commit -m 'Add amazing filter'`)
4. **Push** to the branch (`git push origin feature/amazing-filter`)
5. Open a **Pull Request**

### Ideas for Contributions

- New filters (watercolor, glitch, ASCII art, etc.)
- Video recording (`cv2.VideoWriter` / MediaRecorder API)
- Virtual camera output (`pyvirtualcam`) for Zoom/Meet
- Fingertip trail effects
- GPU acceleration (`cv2.cuda` / WebGL shaders)
- Mobile touch gesture fallbacks
- Accessibility improvements

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with MediaPipe, OpenCV, and hand waves.
</p>
