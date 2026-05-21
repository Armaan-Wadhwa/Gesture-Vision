# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Class-based OpenCV + MediaPipe application for real-time hand-gesture-controlled camera filters. Uses MediaPipe Hands, Selfie Segmentation, and Face Mesh.

Entry: `hand-ditaction.py` (single file, `GestureVision` class). `scratch/` holds throwaway experiments — not part of the main app.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python hand-ditaction.py [--camera N] [--width W] [--height H]
```

Quit with `q`. Needs webcam at specified index (default 0).

## Architecture

### Filter pipeline
- `apply_filter(frame, name, intensity)` — single filter application with intensity blending.
- `apply_stacked(frame, intensity)` — chains multiple toggled filters in index order.
- `_get_filtered()` — dispatcher: uses stack if non-empty, else single filter.
- Region methods (`apply_filter_in_box`, `apply_background_filter`, `apply_face_filter`, `apply_draw_filter`, `apply_magnifier_in_box`) create a mask, apply filter to full frame, then composite masked region onto original.

### Handedness mirroring
Camera frame is flipped (`cv2.flip(frame, 1)`). MediaPipe `"Left"` label = user's actual right hand, `"Right"` = user's left hand. This swap is used throughout gesture detection and hand assignment — preserve it when editing.

### Gesture priority
Gestures are checked in this order per hand per frame (first match wins): pinch → thumbs up/down → fist → open palm → finger count → swipe. Each gesture type has independent cooldown via `gesture_ready()`.

### Modes
NORMAL | BACKGROUND | FACE | DRAW | MAGNIFIER — switched by keyboard. Mode determines which region method is called in the main loop.

### State
All mutable state lives on `GestureVision` instance: current filter index, mode, intensity, frozen frame, draw points, stacked filter set, swipe history, gesture timestamps.

## Planned features
- Cartoon (bilateral + edges), pixelate, blur/bokeh, edge-only, sepia, HSV hue-shift slider — DONE
- Background-only filter via Selfie Segmentation — DONE
- Face-only filter via Face Mesh — DONE
- Gestures: open palm reset, fist freeze, finger-count select, swipe, thumbs up/down — DONE
- Free-draw mask region — DONE
- Magnifier mode — DONE
- Stacked filters — DONE
- Adjustable filter intensity from thumb distance — DONE

### Future ideas
- Record video to mp4 (`cv2.VideoWriter`)
- Stream to virtual cam (`pyvirtualcam`) for Zoom/Meet
- Trail effect on fingertip
- GPU acceleration via `cv2.cuda`
- CLI filter presets

## No tests, no lint config, no build step.
