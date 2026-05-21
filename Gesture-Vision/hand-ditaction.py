#!/usr/bin/env python3
"""Gesture Vision — hand-gesture-controlled camera filters."""

import math
import time
import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from datetime import datetime


class GestureVision:
    FILTERS = [
        "None", "GRAY", "THERMAL", "INVERT", "SKETCH", "VINTAGE",
        "CARTOON", "PIXELATE", "BLUR", "EDGE", "SEPIA", "HSV_SHIFT",
    ]

    MODES = ["NORMAL", "BACKGROUND", "FACE", "DRAW", "MAGNIFIER"]

    GESTURE_COOLDOWN = 0.8
    SWIPE_THRESHOLD = 0.15
    SWIPE_TIME_WINDOW = 0.4
    PINCH_THRESHOLD = 50

    def __init__(self, camera_index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        try:
            self.selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        except Exception:
            self.selfie_seg = None
            print("Warning: Selfie Segmentation unavailable")

        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self.face_mesh = None
            print("Warning: Face Mesh unavailable")

        self.current_filter = 0
        self.mode = "NORMAL"
        self.filter_intensity = 1.0
        self.hue_shift = 30
        self.frozen = False
        self.frozen_frame = None
        self.stacked_filters = set()

        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.last_snapshot_path = None

        self.draw_canvas = None
        self.draw_points = []

        self.last_gesture_time = {}
        self.swipe_history = {}

        self.fps_times = deque(maxlen=30)
        self.fps = 0
        self.finger_count_mode = False

        cv2.namedWindow("Gesture Vision", cv2.WINDOW_NORMAL)

    # ── Helpers ──────────────────────────────────────────────────────

    def get_px(self, landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)

    def gesture_ready(self, name):
        now = time.time()
        if now - self.last_gesture_time.get(name, 0) > self.GESTURE_COOLDOWN:
            self.last_gesture_time[name] = now
            return True
        return False

    # ── Gesture Detection ────────────────────────────────────────────

    def count_fingers(self, hand_lm, label):
        lm = hand_lm.landmark
        count = 0
        if label == "Left":
            if lm[4].x > lm[3].x:
                count += 1
        else:
            if lm[4].x < lm[3].x:
                count += 1
        for tip, pip_ in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip].y < lm[pip_].y:
                count += 1
        return count

    def is_thumb_only(self, hand_lm, label):
        lm = hand_lm.landmark
        if label == "Left":
            thumb_ext = lm[4].x > lm[3].x
        else:
            thumb_ext = lm[4].x < lm[3].x
        others_curled = all(
            lm[t].y > lm[p].y for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)]
        )
        return thumb_ext and others_curled

    def detect_thumbs_direction(self, hand_lm):
        lm = hand_lm.landmark
        if lm[4].y < lm[2].y - 0.05:
            return "up"
        if lm[4].y > lm[2].y + 0.05:
            return "down"
        return None

    def detect_pinch(self, hand_lm, w, h):
        lm = hand_lm.landmark
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)
        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        return math.hypot(ix - tx, iy - ty)

    def detect_swipe(self, hand_lm, hand_id):
        wrist_x = hand_lm.landmark[0].x
        now = time.time()
        if hand_id not in self.swipe_history:
            self.swipe_history[hand_id] = deque(maxlen=15)
        self.swipe_history[hand_id].append((wrist_x, now))
        history = self.swipe_history[hand_id]
        if len(history) < 5:
            return None
        oldest_x, oldest_t = history[0]
        if now - oldest_t > self.SWIPE_TIME_WINDOW:
            dx = wrist_x - oldest_x
            if abs(dx) > self.SWIPE_THRESHOLD:
                self.swipe_history[hand_id].clear()
                return "right" if dx > 0 else "left"
        return None

    # ── Filters ──────────────────────────────────────────────────────

    def apply_filter(self, frame, name, intensity=1.0):
        original = frame.copy()
        if name == "None":
            return original

        if name == "GRAY":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif name == "THERMAL":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        elif name == "INVERT":
            result = cv2.bitwise_not(frame)

        elif name == "SKETCH":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            invblur = 255 - blur
            sketch = cv2.divide(gray, invblur, scale=256.0)
            result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        elif name == "VINTAGE":
            kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189],
            ])
            result = np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)

        elif name == "CARTOON":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 9, 9,
            )
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            result = cv2.bitwise_and(color, color, mask=edges)

        elif name == "PIXELATE":
            ph, pw = frame.shape[:2]
            ps = max(4, min(pw, ph) // 40)
            small = cv2.resize(frame, (pw // ps, ph // ps),
                               interpolation=cv2.INTER_LINEAR)
            result = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)

        elif name == "BLUR":
            result = cv2.GaussianBlur(frame, (45, 45), 0)

        elif name == "EDGE":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif name == "SEPIA":
            kernel = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ])
            result = np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)
            rows, cols = result.shape[:2]
            X = cv2.getGaussianKernel(cols, cols * 0.5)
            Y = cv2.getGaussianKernel(rows, rows * 0.5)
            vignette = Y * X.T
            vignette = vignette / vignette.max()
            for i in range(3):
                result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)

        elif name == "HSV_SHIFT":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + self.hue_shift) % 180
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        else:
            result = original

        if intensity < 1.0 and name != "None":
            result = cv2.addWeighted(original, 1.0 - intensity, result, intensity, 0)

        return result

    def apply_stacked(self, frame, intensity=1.0):
        result = frame.copy()
        for idx in sorted(self.stacked_filters):
            if idx < len(self.FILTERS):
                result = self.apply_filter(result, self.FILTERS[idx], intensity)
        return result

    def _get_filtered(self, frame, filter_name, intensity):
        if self.stacked_filters:
            return self.apply_stacked(frame, intensity)
        return self.apply_filter(frame, filter_name, intensity)

    # ── Region-based application ─────────────────────────────────────

    def apply_filter_in_box(self, frame, left_lm, right_lm, w, h,
                            filter_name, intensity=1.0):
        tl = self.get_px(left_lm.landmark[4], w, h)
        tr = self.get_px(right_lm.landmark[4], w, h)
        br = self.get_px(right_lm.landmark[8], w, h)
        bl = self.get_px(left_lm.landmark[8], w, h)
        pts = np.array([tl, tr, br, bl], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        filtered = self._get_filtered(frame, filter_name, intensity)
        mask3 = cv2.merge([mask, mask, mask])
        result = cv2.add(
            cv2.bitwise_and(filtered, mask3),
            cv2.bitwise_and(frame, cv2.bitwise_not(mask3)),
        )

        cv2.polylines(result, [pts], True, (0, 255, 255), 2)
        for pt, lbl, (ox, oy) in [
            (tl, "L4", (-20, -12)), (tr, "R4", (6, -12)),
            (bl, "L8", (-20, 18)),  (br, "R8", (6, 18)),
        ]:
            cv2.circle(result, pt, 8, (0, 0, 255), -1)
            cv2.putText(result, lbl, (pt[0]+ox, pt[1]+oy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        return result

    def apply_magnifier_in_box(self, frame, left_lm, right_lm, w, h):
        tl = self.get_px(left_lm.landmark[4], w, h)
        tr = self.get_px(right_lm.landmark[4], w, h)
        br = self.get_px(right_lm.landmark[8], w, h)
        bl = self.get_px(left_lm.landmark[8], w, h)
        pts = np.array([tl, tr, br, bl], dtype=np.int32)

        x, y, bw, bh = cv2.boundingRect(pts)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return frame

        rw, rh = (x2 - x1) // 2, (y2 - y1) // 2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx1, ry1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        rx2, ry2 = min(w, rx1 + rw), min(h, ry1 + rh)
        if rx2 - rx1 < 5 or ry2 - ry1 < 5:
            return frame

        roi = frame[ry1:ry2, rx1:rx2]
        zoomed = cv2.resize(roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask3 = cv2.merge([mask, mask, mask])

        temp = frame.copy()
        zh, zw = zoomed.shape[:2]
        if y1 + zh <= h and x1 + zw <= w:
            temp[y1:y1+zh, x1:x1+zw] = zoomed

        result = cv2.add(
            cv2.bitwise_and(temp, mask3),
            cv2.bitwise_and(frame, cv2.bitwise_not(mask3)),
        )
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
        cv2.putText(result, "MAGNIFIER", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return result

    def apply_background_filter(self, frame, filter_name, intensity=1.0):
        if self.selfie_seg is None:
            return frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg = self.selfie_seg.process(rgb)
        person = seg.segmentation_mask > 0.5
        filtered = self._get_filtered(frame, filter_name, intensity)
        output = np.where(person[:, :, None], frame, filtered)
        return output.astype(np.uint8)

    def apply_face_filter(self, frame, filter_name, intensity=1.0):
        if self.face_mesh is None:
            return frame
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return frame

        mask = np.zeros((fh, fw), dtype=np.uint8)
        for face_lm in result.multi_face_landmarks:
            points = [(int(lm.x * fw), int(lm.y * fh)) for lm in face_lm.landmark]
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 255)

        filtered = self._get_filtered(frame, filter_name, intensity)
        mask3 = cv2.merge([mask, mask, mask])
        return cv2.add(
            cv2.bitwise_and(filtered, mask3),
            cv2.bitwise_and(frame, cv2.bitwise_not(mask3)),
        )

    def apply_draw_filter(self, frame, filter_name, intensity=1.0):
        fh, fw = frame.shape[:2]
        if not self.draw_points or len(self.draw_points) < 3:
            return frame
        mask = np.zeros((fh, fw), dtype=np.uint8)
        pts = np.array(self.draw_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        filtered = self._get_filtered(frame, filter_name, intensity)
        mask3 = cv2.merge([mask, mask, mask])
        result = cv2.add(
            cv2.bitwise_and(filtered, mask3),
            cv2.bitwise_and(frame, cv2.bitwise_not(mask3)),
        )
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
        return result

    # ── Snapshot ──────────────────────────────────────────────────────

    def take_snapshot(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.snapshot_dir, f"snapshot_{ts}.png")
        cv2.imwrite(path, frame)
        self.last_snapshot_path = path
        return path

    def discard_last_snapshot(self):
        if self.last_snapshot_path and os.path.exists(self.last_snapshot_path):
            removed = self.last_snapshot_path
            os.remove(removed)
            self.last_snapshot_path = None
            return removed
        return None

    # ── HUD ──────────────────────────────────────────────────────────

    def draw_hud(self, frame, w, h, gesture_text=""):
        selected = self.FILTERS[self.current_filter]

        cv2.putText(frame, f"Filter: {selected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {self.mode}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        bar_x, bar_y, bar_w = w - 130, 30, 100
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                      (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * self.filter_intensity), bar_y + 10),
                      (0, 255, 0), -1)
        cv2.putText(frame, f"Intensity: {self.filter_intensity:.0%}",
                    (bar_x - 10, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {self.fps}", (w - 120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if self.frozen:
            cv2.putText(frame, "FROZEN", (w // 2 - 50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.stacked_filters:
            names = [self.FILTERS[i] for i in sorted(self.stacked_filters)
                     if i < len(self.FILTERS)]
            cv2.putText(frame, "Stack: " + " + ".join(names), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)

        if self.finger_count_mode:
            cv2.putText(frame, "[FINGER-COUNT MODE]", (w // 2 - 100, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if gesture_text:
            cv2.putText(frame, gesture_text, (10, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        lines = [
            "Keys: F=Normal B=Background A=Face D=Draw M=Magnify "
            "N=FingerCount C=Clear S=Save +/-=Hue Q=Quit",
            "Gestures: Pinch=Switch  Palm=Reset  Fist=Freeze  "
            "ThumbUp=Save  ThumbDown=Discard  Swipe=Prev/Next",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - 30 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)
        return frame

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self):
        gesture_feedback = ""
        gesture_fb_time = 0

        while True:
            now = time.time()
            self.fps_times.append(now)
            if len(self.fps_times) > 1:
                elapsed = self.fps_times[-1] - self.fps_times[0]
                self.fps = int(len(self.fps_times) / elapsed) if elapsed > 0 else 0

            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if self.draw_canvas is None:
                self.draw_canvas = np.zeros((h, w), dtype=np.uint8)

            if self.frozen and self.frozen_frame is not None:
                frame = self.frozen_frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb)

            left_hand = None
            right_hand = None
            current_gesture = ""

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_lm, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    label = handedness.classification[0].label
                    if label == "Left":
                        right_hand = hand_lm
                    else:
                        left_hand = hand_lm

                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    )

                    pinch_dist = self.detect_pinch(hand_lm, w, h)
                    fingers = self.count_fingers(hand_lm, label)
                    swipe = self.detect_swipe(hand_lm, label)

                    if pinch_dist < self.PINCH_THRESHOLD:
                        if label == "Left" and self.gesture_ready("pinch_next"):
                            self.current_filter = (
                                (self.current_filter + 1) % len(self.FILTERS)
                            )
                            current_gesture = "NEXT FILTER"
                        elif label == "Right" and self.gesture_ready("pinch_prev"):
                            self.current_filter = (
                                (self.current_filter - 1) % len(self.FILTERS)
                            )
                            current_gesture = "PREV FILTER"

                    elif self.is_thumb_only(hand_lm, label):
                        direction = self.detect_thumbs_direction(hand_lm)
                        if direction == "up" and self.gesture_ready("thumbs_up"):
                            path = self.take_snapshot(frame)
                            current_gesture = f"SAVED {os.path.basename(path)}"
                        elif direction == "down" and self.gesture_ready("thumbs_down"):
                            removed = self.discard_last_snapshot()
                            current_gesture = (
                                f"DISCARDED {os.path.basename(removed)}"
                                if removed else "NOTHING TO DISCARD"
                            )

                    elif fingers == 0 and self.gesture_ready("fist"):
                        self.frozen = not self.frozen
                        if self.frozen:
                            self.frozen_frame = frame.copy()
                            current_gesture = "FRAME FROZEN"
                        else:
                            self.frozen_frame = None
                            current_gesture = "FRAME UNFROZEN"

                    elif fingers == 5 and self.gesture_ready("open_palm"):
                        self.current_filter = 0
                        self.stacked_filters.clear()
                        self.filter_intensity = 1.0
                        current_gesture = "RESET"

                    elif self.finger_count_mode and 1 <= fingers <= 4:
                        if self.gesture_ready("finger_select"):
                            if fingers < len(self.FILTERS):
                                self.current_filter = fingers
                                current_gesture = (
                                    f"FINGER #{fingers}: "
                                    f"{self.FILTERS[fingers]}"
                                )

                    if swipe and not current_gesture:
                        if swipe == "right" and self.gesture_ready("swipe"):
                            self.current_filter = (
                                (self.current_filter + 1) % len(self.FILTERS)
                            )
                            current_gesture = "SWIPE -> NEXT"
                        elif swipe == "left" and self.gesture_ready("swipe"):
                            self.current_filter = (
                                (self.current_filter - 1) % len(self.FILTERS)
                            )
                            current_gesture = "SWIPE <- PREV"

                    if self.mode == "DRAW" and label == "Right":
                        if fingers >= 1:
                            px, py = self.get_px(hand_lm.landmark[8], w, h)
                            self.draw_points.append((px, py))
                            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            if left_hand and right_hand:
                lt = left_hand.landmark[4]
                rt = right_hand.landmark[4]
                thumb_dist = math.hypot((lt.x - rt.x) * w, (lt.y - rt.y) * h)
                self.filter_intensity = float(np.clip(thumb_dist / 400.0, 0.0, 1.0))

            selected = self.FILTERS[self.current_filter]

            try:
                if self.mode == "NORMAL":
                    if left_hand and right_hand:
                        frame = self.apply_filter_in_box(
                            frame, left_hand, right_hand, w, h,
                            selected, self.filter_intensity,
                        )
                    elif self.stacked_filters:
                        frame = self.apply_stacked(frame, self.filter_intensity)
                    else:
                        frame = self.apply_filter(
                            frame, selected, self.filter_intensity,
                        )
                elif self.mode == "BACKGROUND":
                    frame = self.apply_background_filter(
                        frame, selected, self.filter_intensity,
                    )
                elif self.mode == "FACE":
                    frame = self.apply_face_filter(
                        frame, selected, self.filter_intensity,
                    )
                elif self.mode == "DRAW":
                    frame = self.apply_draw_filter(
                        frame, selected, self.filter_intensity,
                    )
                elif self.mode == "MAGNIFIER":
                    if left_hand and right_hand:
                        frame = self.apply_magnifier_in_box(
                            frame, left_hand, right_hand, w, h,
                        )
            except Exception as e:
                cv2.putText(frame, f"Filter error: {str(e)[:60]}", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if current_gesture:
                gesture_feedback = current_gesture
                gesture_fb_time = time.time()
            elif time.time() - gesture_fb_time > 2.0:
                gesture_feedback = ""

            frame = self.draw_hud(frame, w, h, gesture_feedback)
            cv2.imshow("Gesture Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                self.mode = "NORMAL"
            elif key == ord("b"):
                self.mode = "BACKGROUND"
            elif key == ord("a"):
                self.mode = "FACE"
            elif key == ord("d"):
                self.mode = "DRAW"
            elif key == ord("m"):
                self.mode = "MAGNIFIER"
            elif key == ord("c"):
                self.draw_points.clear()
                self.draw_canvas = np.zeros((h, w), dtype=np.uint8)
            elif key == ord("s"):
                path = self.take_snapshot(frame)
                gesture_feedback = f"SAVED {os.path.basename(path)}"
                gesture_fb_time = time.time()
            elif key == ord("n"):
                self.finger_count_mode = not self.finger_count_mode
                gesture_feedback = (
                    f"Finger Count Mode: "
                    f"{'ON' if self.finger_count_mode else 'OFF'}"
                )
                gesture_fb_time = time.time()
            elif key in (ord("+"), ord("=")):
                self.hue_shift = (self.hue_shift + 10) % 180
            elif key == ord("-"):
                self.hue_shift = (self.hue_shift - 10) % 180
            elif ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < len(self.FILTERS):
                    self.stacked_filters.symmetric_difference_update({idx})
                    gesture_feedback = f"Stack toggled: {self.FILTERS[idx]}"
                    gesture_fb_time = time.time()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gesture Vision — hand gesture controlled camera filters",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    args = parser.parse_args()

    app = GestureVision(
        camera_index=args.camera, width=args.width, height=args.height,
    )
    app.run()
