# Gesture Vision — Tutorial

A step-by-step guide to every feature in Gesture Vision.

---

## 1. Getting Started

```bash
source .venv/bin/activate
python hand-ditaction.py
```

A window titled **Gesture Vision** opens showing your webcam feed. The HUD at the top shows the current filter, mode, intensity bar, and FPS. Controls are listed at the bottom.

---

## 2. Filters

### How filters work

Each filter transforms the pixel data of the camera frame. You cycle through filters with pinch gestures or keyboard shortcuts.

### Filter reference

**None** — No processing. Raw camera feed.

**GRAY** — Converts each frame to grayscale, then back to BGR so it displays in the same window. Useful as a base for stacking.

**THERMAL** — Converts to grayscale, then applies OpenCV's JET colormap. Cool areas appear blue, warm areas red — simulates a thermal camera.

**INVERT** — Bitwise NOT on every pixel. White becomes black, red becomes cyan, etc.

**SKETCH** — Converts to grayscale, inverts, blurs the inverted image, inverts again, then divides the original gray by the result. Produces a pencil-drawing look.

**VINTAGE** — Multiplies each pixel by a 3×3 color-transform matrix that shifts tones toward warm yellow-brown, mimicking aged film.

**CARTOON** — Two-step process: (1) adaptive thresholding on a median-blurred grayscale copy produces bold black edges, (2) bilateral filtering smooths the color image while keeping edges sharp. The edge mask is AND-ed with the smooth color to produce the final cartoon look.

**PIXELATE** — Downscales the frame to roughly 1/40th resolution, then upscales back using nearest-neighbor interpolation. Creates a blocky mosaic effect.

**BLUR** — Applies a large (45×45) Gaussian blur kernel. Simulates a soft bokeh / out-of-focus look.

**EDGE** — Canny edge detection on grayscale. Only edges are drawn as white lines on black — everything else disappears.

**SEPIA** — Similar color-transform matrix to Vintage, but with a vignette overlay: a 2D Gaussian darkens the edges of the frame and keeps the center bright, giving an old-photograph feel.

**HSV_SHIFT** — Converts to HSV color space, adds an offset to the Hue channel, converts back. The offset is adjustable with `+` / `-` keys (default shift: 30). Produces psychedelic color rotations.

---

## 3. Modes

Press a key to switch modes. The current mode is shown in the HUD.

### NORMAL (key: F)

Default mode. Behavior depends on how many hands are visible:

- **No hands**: filter is applied to the entire frame.
- **One hand**: filter is applied to the entire frame; gesture detection runs on that hand.
- **Two hands**: filter is applied **only inside the quadrilateral** formed by:
  - Top-left = left thumb tip (landmark 4)
  - Top-right = right thumb tip (landmark 4)
  - Bottom-right = right index tip (landmark 8)
  - Bottom-left = left index tip (landmark 8)

  Outside the box, the original feed is shown. A yellow border and corner labels (L4, R4, L8, R8) mark the box.

### BACKGROUND (key: B)

Uses MediaPipe Selfie Segmentation to separate you from the background. The filter is applied **only to the background** — your body stays unfiltered. Works with any filter.

If Selfie Segmentation failed to load at startup (missing model), this mode silently falls back to showing the raw frame.

### FACE (key: A)

Uses MediaPipe Face Mesh to detect your face. A convex hull around all 468 face landmarks creates a mask. The filter is applied **only inside that mask** — the rest of the frame is untouched.

If no face is detected, the frame passes through unchanged.

### DRAW (key: D)

Freehand drawing mode. Your **left hand's index finger** (MediaPipe label "Right" due to mirror flip) acts as a pen:

1. Extend your index finger and move it around — green dots trace your path on screen.
2. The path automatically closes into a polygon.
3. The current filter fills the polygon interior.
4. Press **C** to clear the canvas and start over.

Tips:
- Draw slowly for smooth curves.
- The polygon closes from the last point back to the first, so draw a complete outline.
- Combine with filter stacking for creative effects inside the drawn region.

### MAGNIFIER (key: M)

Requires both hands. Instead of applying a filter, the box region is **zoomed in 2×**. The center half of the box is cropped and scaled up to fill the full box area, creating a magnifying-glass effect. A green border marks the zoom region.

---

## 4. Gestures

All gestures have a cooldown (0.8 seconds) to prevent repeated triggers. The camera frame is mirrored, so MediaPipe's "Left" label corresponds to your actual right hand and vice versa.

### Pinch (thumb tip + index tip close together)

- **Right hand pinch** → next filter in the list.
- **Left hand pinch** → previous filter.
- Detection: distance between landmarks 4 and 8 falls below 50 pixels.

### Open Palm (all 5 fingers extended)

- Resets filter to "None", clears any stacked filters, and restores intensity to 100%.
- Works with either hand.

### Fist (0 fingers extended)

- Toggles **freeze frame**. When frozen, the display shows the captured frame and "FROZEN" appears in the HUD. Make a fist again to unfreeze and resume live feed.

### Thumbs Up (only thumb extended, pointing upward)

- Saves the current frame as a PNG in the `snapshots/` directory. Filename includes a timestamp.
- The HUD briefly shows the saved filename.

### Thumbs Down (only thumb extended, pointing downward)

- Deletes the most recently saved snapshot from disk.
- If no snapshot exists, shows "NOTHING TO DISCARD".

### Horizontal Swipe (rapid wrist movement)

- Swipe right → next filter.
- Swipe left → previous filter.
- Detection: wrist (landmark 0) moves more than 15% of the frame width within 0.4 seconds.
- Lower priority than pinch — if a pinch is detected in the same frame, swipe is ignored.

### Two-Hand Intensity Control

- When both hands are visible, the distance between the two **thumb tips** controls filter intensity.
- Hands close together → low intensity (nearly original). Hands far apart → full intensity.
- The green intensity bar in the HUD updates in real time.

---

## 5. Finger-Count Mode

Toggle with the **N** key. When active, "[FINGER-COUNT MODE]" appears in the HUD.

Show 1–4 fingers on either hand to jump directly to that filter index:
- 1 finger → GRAY
- 2 fingers → THERMAL
- 3 fingers → INVERT
- 4 fingers → SKETCH

0 fingers (fist) and 5 fingers (open palm) keep their normal gesture meanings (freeze and reset). Finger-count selection has a cooldown to avoid rapid switching.

---

## 6. Filter Stacking

Press number keys **0–9** to toggle individual filters into a stack. When the stack is non-empty, it overrides the single-filter selection — all stacked filters are applied in index order.

Example: press `1` then `3` to stack GRAY + INVERT. The HUD shows "Stack: GRAY + INVERT". Press the same key again to remove a filter from the stack.

Stacking works in all modes (normal, background, face, draw). Press **open palm** gesture or manually clear the stack by pressing each key again.

---

## 7. Keyboard Reference

| Key | Action |
|-----|--------|
| Q | Quit |
| F | Normal mode |
| B | Background mode |
| A | Face mode |
| D | Draw mode |
| M | Magnifier mode |
| N | Toggle finger-count mode |
| C | Clear drawn region |
| S | Save snapshot |
| + / = | Increase HSV hue shift by 10 |
| - | Decrease HSV hue shift by 10 |
| 0–9 | Toggle filter in/out of stack |

---

## 8. CLI Options

```
python hand-ditaction.py --camera 1         # use camera index 1
python hand-ditaction.py --width 640 --height 480   # lower resolution
```

Lower resolution = faster processing = higher FPS, especially on older hardware.

---

## 9. Troubleshooting

**"Error: Cannot open camera 0"** — No webcam found at index 0. Try `--camera 1` or check your webcam connection.

**Low FPS** — Background and Face modes run extra ML models per frame. Switch to Normal mode or reduce resolution.

**Gestures not detected** — Make sure your hand is well-lit, fully in frame, and not too close to the camera. MediaPipe needs a detection confidence ≥ 0.7.

**Filters look wrong in stacked mode** — Some filter combinations don't compose well (e.g. EDGE + BLUR cancels the edges). Experiment with different stacks.

**Selfie Segmentation / Face Mesh warning at startup** — The MediaPipe model files may not have downloaded correctly. Reinstall mediapipe: `pip install --force-reinstall mediapipe`.
