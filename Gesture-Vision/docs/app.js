/* Gesture Vision — browser edition */

const FILTERS = [
  "None","GRAY","THERMAL","INVERT","SKETCH","VINTAGE",
  "CARTOON","PIXELATE","BLUR","EDGE","SEPIA","HSV_SHIFT",
];
const MODES = ["NORMAL","BACKGROUND","FACE","DRAW","MAGNIFIER"];

const HAND_CONNS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const GESTURE_CD = 800;
const PINCH_TH   = 0.07;
const SWIPE_TH   = 0.15;
const SWIPE_WIN  = 400;

// ── Helpers ────────────────────────────────────────────────────

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0, s = mx ? d / mx : 0, v = mx;
  if (d) {
    if (mx === r)      h = ((g - b) / d + 6) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else               h = (r - g) / d + 4;
    h *= 60;
  }
  return [h, s, v];
}
function hsvToRgb(h, s, v) {
  const c = v * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = v - c;
  let r, g, b;
  if      (h < 60)  { r=c; g=x; b=0; }
  else if (h < 120) { r=x; g=c; b=0; }
  else if (h < 180) { r=0; g=c; b=x; }
  else if (h < 240) { r=0; g=x; b=c; }
  else if (h < 300) { r=x; g=0; b=c; }
  else              { r=c; g=0; b=x; }
  return [(r+m)*255, (g+m)*255, (b+m)*255];
}
function jetColor(t) {
  if (t < 0.25)     return [0, t*4*255, 255];
  else if (t < 0.5) return [0, 255, (1-(t-0.25)*4)*255];
  else if (t < 0.75)return [(t-0.5)*4*255, 255, 0];
  else              return [255, (1-(t-0.75)*4)*255, 0];
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function lmPx(lm, w, h) {
  return [lm.x * w, lm.y * h];
}

// ── App ────────────────────────────────────────────────────────

class GestureVision {
  constructor() {
    this.video  = document.getElementById("video");
    this.canvas = document.getElementById("canvas");
    this.ctx    = this.canvas.getContext("2d", { willReadFrequently: true });

    this._tmpC = document.createElement("canvas");
    this._tmpX = this._tmpC.getContext("2d", { willReadFrequently: true });
    this._blrC = document.createElement("canvas");
    this._blrX = this._blrC.getContext("2d");

    this.filter = 0;
    this.mode   = "NORMAL";
    this.intensity = 1.0;
    this.hueShift  = 30;
    this.frozen    = false;
    this.frozenData = null;

    this.drawPoints = [];
    this.lastGesture = {};
    this.swipeHist   = {};
    this.fpsTimes    = [];
    this.fps = 0;
    this.lastTs = 0;

    this.handLandmarker = null;
    this.segmenter      = null;
    this.faceLandmarker  = null;
    this.segLoading = false;
    this.faceLoading = false;
    this.visionFiles = null;

    this.snapshots = [];
  }

  async init() {
    const status = document.getElementById("load-status");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
      });
      this.video.srcObject = stream;
      await this.video.play();
    } catch (e) {
      document.getElementById("load-error").textContent =
        "Camera access denied or unavailable. " + e.message;
      document.getElementById("load-error").hidden = false;
      return;
    }

    const w = this.video.videoWidth, h = this.video.videoHeight;
    this.canvas.width = this._tmpC.width = this._blrC.width = w;
    this.canvas.height = this._tmpC.height = this._blrC.height = h;

    status.textContent = "Loading hand detection model...";

    try {
      this.visionFiles = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      this.handLandmarker = await HandLandmarker.createFromOptions(this.visionFiles, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    } catch (e) {
      status.textContent = "Failed to load hand model: " + e.message;
      document.getElementById("load-error").hidden = false;
      return;
    }

    this.buildUI();
    document.getElementById("loading").hidden = true;
    document.getElementById("app").hidden = false;

    this.processFrame();
  }

  // ── Lazy model loading ───────────────────────────────────────

  async ensureSegmenter() {
    if (this.segmenter || this.segLoading) return;
    this.segLoading = true;
    this.toast("Loading background model...");
    try {
      this.segmenter = await ImageSegmenter.createFromOptions(this.visionFiles, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        outputConfidenceMasks: true,
      });
    } catch (e) {
      this.toast("Background model failed");
      this.mode = "NORMAL";
    }
    this.segLoading = false;
  }

  async ensureFaceLandmarker() {
    if (this.faceLandmarker || this.faceLoading) return;
    this.faceLoading = true;
    this.toast("Loading face model...");
    try {
      this.faceLandmarker = await FaceLandmarker.createFromOptions(this.visionFiles, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });
    } catch (e) {
      this.toast("Face model failed");
      this.mode = "NORMAL";
    }
    this.faceLoading = false;
  }

  // ── UI ───────────────────────────────────────────────────────

  buildUI() {
    const sel = document.getElementById("sel-filter");
    FILTERS.forEach((f, i) => {
      const opt = document.createElement("option");
      opt.value = i; opt.textContent = f;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => { this.filter = +sel.value; });

    const modeDiv = document.getElementById("mode-btns");
    MODES.forEach(m => {
      const btn = document.createElement("button");
      btn.textContent = m[0] + m.slice(1).toLowerCase();
      btn.dataset.mode = m;
      if (m === this.mode) btn.classList.add("active");
      btn.addEventListener("click", () => this.setMode(m));
      modeDiv.appendChild(btn);
    });

    document.getElementById("intensity").addEventListener("input", e => {
      this.intensity = e.target.value / 100;
      document.getElementById("intensity-val").textContent = e.target.value + "%";
    });
    document.getElementById("hue-shift").addEventListener("input", e => {
      this.hueShift = +e.target.value;
      document.getElementById("hue-val").textContent = e.target.value;
    });

    document.getElementById("btn-snapshot").addEventListener("click", () => this.takeSnapshot());
    document.getElementById("btn-freeze").addEventListener("click", () => this.toggleFreeze());
    document.getElementById("btn-clear").addEventListener("click", () => { this.drawPoints = []; });

    document.addEventListener("keydown", e => this.onKey(e));
  }

  setMode(m) {
    this.mode = m;
    if (m === "BACKGROUND") this.ensureSegmenter();
    if (m === "FACE") this.ensureFaceLandmarker();
    document.querySelectorAll("#mode-btns button").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === m);
    });
  }

  onKey(e) {
    const k = e.key.toLowerCase();
    if (k === "f") this.setMode("NORMAL");
    else if (k === "b") this.setMode("BACKGROUND");
    else if (k === "a") this.setMode("FACE");
    else if (k === "d") this.setMode("DRAW");
    else if (k === "m") this.setMode("MAGNIFIER");
    else if (k === "c") this.drawPoints = [];
    else if (k === "s") this.takeSnapshot();
    else if (k === " ") { e.preventDefault(); this.toggleFreeze(); }
  }

  toast(msg) {
    const el = document.getElementById("gesture-toast");
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => { el.hidden = true; }, 2000);
  }

  // ── Snapshot ─────────────────────────────────────────────────

  takeSnapshot() {
    const url = this.canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = "gesture_vision_" + Date.now() + ".png";
    a.click();
    this.snapshots.push(url);
    this.toast("Snapshot saved");
  }

  toggleFreeze() {
    this.frozen = !this.frozen;
    if (this.frozen) {
      this.frozenData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    }
    document.getElementById("btn-freeze").classList.toggle("active", this.frozen);
    this.toast(this.frozen ? "FROZEN" : "UNFROZEN");
  }

  // ── Gesture Detection ────────────────────────────────────────

  gestureReady(name) {
    const now = performance.now();
    if (now - (this.lastGesture[name] || 0) > GESTURE_CD) {
      this.lastGesture[name] = now;
      return true;
    }
    return false;
  }

  countFingers(lm, label) {
    let count = 0;
    // Thumb: x-direction depends on handedness (mirrored frame)
    if (label === "Left") {
      if (lm[4].x > lm[3].x) count++;
    } else {
      if (lm[4].x < lm[3].x) count++;
    }
    if (lm[8].y  < lm[6].y)  count++;
    if (lm[12].y < lm[10].y) count++;
    if (lm[16].y < lm[14].y) count++;
    if (lm[20].y < lm[18].y) count++;
    return count;
  }

  isThumbOnly(lm, label) {
    const thumbExt = label === "Left" ? lm[4].x > lm[3].x : lm[4].x < lm[3].x;
    const othersCurled = lm[8].y > lm[6].y && lm[12].y > lm[10].y &&
                         lm[16].y > lm[14].y && lm[20].y > lm[18].y;
    return thumbExt && othersCurled;
  }

  detectSwipe(lm, id) {
    const wx = lm[0].x, now = performance.now();
    if (!this.swipeHist[id]) this.swipeHist[id] = [];
    this.swipeHist[id].push({ x: wx, t: now });
    if (this.swipeHist[id].length > 15) this.swipeHist[id].shift();
    const h = this.swipeHist[id];
    if (h.length < 5) return null;
    const oldest = h[0];
    if (now - oldest.t > SWIPE_WIN) {
      const dx = wx - oldest.x;
      if (Math.abs(dx) > SWIPE_TH) {
        this.swipeHist[id] = [];
        return dx > 0 ? "right" : "left";
      }
    }
    return null;
  }

  processGestures(handResults) {
    let leftHand = null, rightHand = null;
    let gesture = "";

    if (!handResults.landmarks || handResults.landmarks.length === 0)
      return { leftHand, rightHand, gesture };

    for (let i = 0; i < handResults.landmarks.length; i++) {
      const lm = handResults.landmarks[i];
      const label = handResults.handedness[i][0].categoryName;

      // Mirrored: MediaPipe "Left" = user's right hand
      if (label === "Left") rightHand = lm;
      else leftHand = lm;

      const pinch = dist(lm[4], lm[8]);
      const fingers = this.countFingers(lm, label);
      const swipe = this.detectSwipe(lm, label);

      if (pinch < PINCH_TH) {
        if (label === "Left" && this.gestureReady("pinch_next")) {
          this.filter = (this.filter + 1) % FILTERS.length;
          gesture = "NEXT: " + FILTERS[this.filter];
        } else if (label === "Right" && this.gestureReady("pinch_prev")) {
          this.filter = (this.filter - 1 + FILTERS.length) % FILTERS.length;
          gesture = "PREV: " + FILTERS[this.filter];
        }
      } else if (this.isThumbOnly(lm, label)) {
        if (lm[4].y < lm[2].y - 0.05 && this.gestureReady("thumb_up")) {
          this.takeSnapshot();
          gesture = "SNAPSHOT SAVED";
        } else if (lm[4].y > lm[2].y + 0.05 && this.gestureReady("thumb_down")) {
          gesture = "THUMBS DOWN";
        }
      } else if (fingers === 0 && this.gestureReady("fist")) {
        this.toggleFreeze();
        gesture = this.frozen ? "FROZEN" : "UNFROZEN";
      } else if (fingers === 5 && this.gestureReady("palm")) {
        this.filter = 0;
        this.intensity = 1.0;
        document.getElementById("intensity").value = 100;
        document.getElementById("intensity-val").textContent = "100%";
        gesture = "RESET";
      }

      if (swipe && !gesture) {
        if (swipe === "right" && this.gestureReady("swipe")) {
          this.filter = (this.filter + 1) % FILTERS.length;
          gesture = "SWIPE NEXT";
        } else if (swipe === "left" && this.gestureReady("swipe")) {
          this.filter = (this.filter - 1 + FILTERS.length) % FILTERS.length;
          gesture = "SWIPE PREV";
        }
      }

      // Draw mode: left hand index finger draws
      if (this.mode === "DRAW" && label === "Right" && fingers >= 1) {
        const w = this.canvas.width, h = this.canvas.height;
        this.drawPoints.push([lm[8].x * w, lm[8].y * h]);
      }
    }

    // Two-hand intensity
    if (leftHand && rightHand) {
      const d = dist(leftHand[4], rightHand[4]);
      this.intensity = Math.min(1, Math.max(0, d / 0.4));
      document.getElementById("intensity").value = Math.round(this.intensity * 100);
      document.getElementById("intensity-val").textContent = Math.round(this.intensity * 100) + "%";
    }

    if (gesture) {
      this.toast(gesture);
      document.getElementById("sel-filter").value = this.filter;
    }

    return { leftHand, rightHand, gesture };
  }

  // ── Filters ──────────────────────────────────────────────────

  applyFilter(imgData, name, intensity) {
    if (name === "None" || intensity === 0) return imgData;
    const d = imgData.data;
    const len = d.length;
    const orig = intensity < 1 ? new Uint8ClampedArray(d) : null;

    switch (name) {
      case "GRAY":
        for (let i = 0; i < len; i += 4) {
          const g = 0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2];
          d[i] = d[i+1] = d[i+2] = g;
        } break;

      case "THERMAL":
        for (let i = 0; i < len; i += 4) {
          const g = (0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2]) / 255;
          const [r, gr, b] = jetColor(g);
          d[i] = r; d[i+1] = gr; d[i+2] = b;
        } break;

      case "INVERT":
        for (let i = 0; i < len; i += 4) {
          d[i] = 255-d[i]; d[i+1] = 255-d[i+1]; d[i+2] = 255-d[i+2];
        } break;

      case "SKETCH": {
        const w = imgData.width, h = imgData.height;
        const gray = new Uint8Array(w * h);
        for (let i = 0; i < gray.length; i++)
          gray[i] = 0.299*d[i*4] + 0.587*d[i*4+1] + 0.114*d[i*4+2];

        const invD = new ImageData(w, h);
        for (let i = 0; i < gray.length; i++) {
          const v = 255 - gray[i];
          invD.data[i*4] = invD.data[i*4+1] = invD.data[i*4+2] = v;
          invD.data[i*4+3] = 255;
        }
        this._tmpX.putImageData(invD, 0, 0);
        this._blrX.filter = "blur(10px)";
        this._blrX.drawImage(this._tmpC, 0, 0);
        this._blrX.filter = "none";
        const bl = this._blrX.getImageData(0, 0, w, h).data;

        for (let i = 0; i < gray.length; i++) {
          const ib = 255 - bl[i*4];
          const val = ib < 1 ? 255 : Math.min(255, (gray[i] * 256) / ib);
          d[i*4] = d[i*4+1] = d[i*4+2] = val;
        }
        break;
      }

      case "VINTAGE":
        for (let i = 0; i < len; i += 4) {
          const r=d[i], g=d[i+1], b=d[i+2];
          d[i]   = Math.min(255, 0.393*r + 0.769*g + 0.189*b);
          d[i+1] = Math.min(255, 0.349*r + 0.686*g + 0.168*b);
          d[i+2] = Math.min(255, 0.272*r + 0.534*g + 0.131*b);
        } break;

      case "CARTOON": {
        const w = imgData.width, h = imgData.height;
        const grayC = new Float32Array(w * h);
        for (let i = 0; i < grayC.length; i++)
          grayC[i] = 0.299*d[i*4] + 0.587*d[i*4+1] + 0.114*d[i*4+2];

        const edges = new Float32Array(w * h);
        for (let y = 1; y < h-1; y++) {
          for (let x = 1; x < w-1; x++) {
            const idx = y*w+x;
            const gx = -grayC[idx-w-1]+grayC[idx-w+1]-2*grayC[idx-1]+2*grayC[idx+1]-grayC[idx+w-1]+grayC[idx+w+1];
            const gy = -grayC[idx-w-1]-2*grayC[idx-w]-grayC[idx-w+1]+grayC[idx+w-1]+2*grayC[idx+w]+grayC[idx+w+1];
            edges[idx] = Math.sqrt(gx*gx+gy*gy);
          }
        }
        for (let i = 0; i < grayC.length; i++) {
          const p = i*4;
          if (edges[i] > 40) { d[p]=d[p+1]=d[p+2]=0; }
          else { d[p]=Math.round(d[p]/40)*40; d[p+1]=Math.round(d[p+1]/40)*40; d[p+2]=Math.round(d[p+2]/40)*40; }
        }
        break;
      }

      case "PIXELATE": {
        const w = imgData.width, h = imgData.height, ps = Math.max(4, Math.min(w,h)/40|0);
        for (let by = 0; by < h; by += ps) {
          for (let bx = 0; bx < w; bx += ps) {
            let sr=0,sg=0,sb=0,cnt=0;
            for (let y=by; y<Math.min(by+ps,h); y++) {
              for (let x=bx; x<Math.min(bx+ps,w); x++) {
                const p=(y*w+x)*4; sr+=d[p]; sg+=d[p+1]; sb+=d[p+2]; cnt++;
              }
            }
            sr/=cnt; sg/=cnt; sb/=cnt;
            for (let y=by; y<Math.min(by+ps,h); y++) {
              for (let x=bx; x<Math.min(bx+ps,w); x++) {
                const p=(y*w+x)*4; d[p]=sr; d[p+1]=sg; d[p+2]=sb;
              }
            }
          }
        }
        break;
      }

      case "BLUR":
        this._tmpX.putImageData(imgData, 0, 0);
        this._blrX.filter = "blur(15px)";
        this._blrX.drawImage(this._tmpC, 0, 0);
        this._blrX.filter = "none";
        const blD = this._blrX.getImageData(0, 0, imgData.width, imgData.height);
        imgData.data.set(blD.data);
        break;

      case "EDGE": {
        const w = imgData.width, h = imgData.height;
        const gE = new Float32Array(w*h);
        for (let i=0;i<gE.length;i++) gE[i]=0.299*d[i*4]+0.587*d[i*4+1]+0.114*d[i*4+2];
        for (let y=1;y<h-1;y++) {
          for (let x=1;x<w-1;x++) {
            const i=y*w+x;
            const gx=-gE[i-w-1]+gE[i-w+1]-2*gE[i-1]+2*gE[i+1]-gE[i+w-1]+gE[i+w+1];
            const gy=-gE[i-w-1]-2*gE[i-w]-gE[i-w+1]+gE[i+w-1]+2*gE[i+w]+gE[i+w+1];
            const v=Math.min(255,Math.sqrt(gx*gx+gy*gy));
            d[i*4]=d[i*4+1]=d[i*4+2]=v;
          }
        }
        break;
      }

      case "SEPIA": {
        const w = imgData.width, h = imgData.height;
        const cx=w/2, cy=h/2, maxD=Math.sqrt(cx*cx+cy*cy);
        for (let y=0;y<h;y++) {
          for (let x=0;x<w;x++) {
            const p=(y*w+x)*4;
            const r=d[p], g=d[p+1], b=d[p+2];
            const dd=Math.sqrt((x-cx)**2+(y-cy)**2);
            const vig=1-0.4*(dd/maxD);
            d[p]  =Math.min(255,(0.393*r+0.769*g+0.189*b)*vig);
            d[p+1]=Math.min(255,(0.349*r+0.686*g+0.168*b)*vig);
            d[p+2]=Math.min(255,(0.272*r+0.534*g+0.131*b)*vig);
          }
        }
        break;
      }

      case "HSV_SHIFT":
        for (let i=0;i<len;i+=4) {
          let [h,s,v] = rgbToHsv(d[i],d[i+1],d[i+2]);
          h = (h + this.hueShift * 2) % 360;
          const [r,g,b] = hsvToRgb(h,s,v);
          d[i]=r; d[i+1]=g; d[i+2]=b;
        } break;
    }

    if (orig) {
      const a = intensity, b = 1 - intensity;
      for (let i = 0; i < len; i += 4) {
        d[i]   = orig[i]*b   + d[i]*a;
        d[i+1] = orig[i+1]*b + d[i+1]*a;
        d[i+2] = orig[i+2]*b + d[i+2]*a;
      }
    }

    return imgData;
  }

  // ── Rendering ────────────────────────────────────────────────

  drawMirroredFrame() {
    const w = this.canvas.width, h = this.canvas.height;
    this.ctx.save();
    this.ctx.translate(w, 0);
    this.ctx.scale(-1, 1);
    this.ctx.drawImage(this.video, 0, 0, w, h);
    this.ctx.restore();
  }

  drawHands(handResults) {
    if (!handResults.landmarks) return;
    const w = this.canvas.width, h = this.canvas.height;

    for (const lm of handResults.landmarks) {
      this.ctx.strokeStyle = "#00ff88";
      this.ctx.lineWidth = 2;
      for (const [a, b] of HAND_CONNS) {
        this.ctx.beginPath();
        this.ctx.moveTo(lm[a].x * w, lm[a].y * h);
        this.ctx.lineTo(lm[b].x * w, lm[b].y * h);
        this.ctx.stroke();
      }
      this.ctx.fillStyle = "#ff3366";
      for (const pt of lm) {
        this.ctx.beginPath();
        this.ctx.arc(pt.x * w, pt.y * h, 3, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }
  }

  renderNormal(leftHand, rightHand) {
    const w = this.canvas.width, h = this.canvas.height;
    const name = FILTERS[this.filter];

    if (leftHand && rightHand && name !== "None") {
      // Filter inside box
      const frameData = this.ctx.getImageData(0, 0, w, h);
      const filtered = this.applyFilter(
        new ImageData(new Uint8ClampedArray(frameData.data), w, h),
        name, this.intensity
      );
      this._tmpX.putImageData(filtered, 0, 0);

      const tl = lmPx(leftHand[4], w, h);
      const tr = lmPx(rightHand[4], w, h);
      const br = lmPx(rightHand[8], w, h);
      const bl = lmPx(leftHand[8], w, h);

      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.moveTo(...tl); this.ctx.lineTo(...tr);
      this.ctx.lineTo(...br); this.ctx.lineTo(...bl);
      this.ctx.closePath();
      this.ctx.clip();
      this.ctx.drawImage(this._tmpC, 0, 0);
      this.ctx.restore();

      // Box border
      this.ctx.strokeStyle = "#00ffff";
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.moveTo(...tl); this.ctx.lineTo(...tr);
      this.ctx.lineTo(...br); this.ctx.lineTo(...bl);
      this.ctx.closePath();
      this.ctx.stroke();
    } else if (name !== "None") {
      const data = this.ctx.getImageData(0, 0, w, h);
      this.applyFilter(data, name, this.intensity);
      this.ctx.putImageData(data, 0, 0);
    }
  }

  renderBackground(timestamp) {
    if (!this.segmenter) return;
    const w = this.canvas.width, h = this.canvas.height;
    const name = FILTERS[this.filter];
    if (name === "None") return;

    let maskData = null;
    try {
      const seg = this.segmenter.segmentForVideo(this.canvas, timestamp);
      if (seg.confidenceMasks && seg.confidenceMasks.length > 0) {
        maskData = seg.confidenceMasks[0].getAsFloat32Array();
      }
    } catch (e) { return; }

    if (!maskData) return;

    const frameData = this.ctx.getImageData(0, 0, w, h);
    const origData = new Uint8ClampedArray(frameData.data);
    this.applyFilter(frameData, name, this.intensity);
    const d = frameData.data;

    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] > 0.5) {
        const p = i * 4;
        d[p] = origData[p]; d[p+1] = origData[p+1]; d[p+2] = origData[p+2];
      }
    }
    this.ctx.putImageData(frameData, 0, 0);
  }

  renderFace(timestamp) {
    if (!this.faceLandmarker) return;
    const w = this.canvas.width, h = this.canvas.height;
    const name = FILTERS[this.filter];
    if (name === "None") return;

    let faceLm = null;
    try {
      const res = this.faceLandmarker.detectForVideo(this.canvas, timestamp);
      if (res.faceLandmarks && res.faceLandmarks.length > 0) faceLm = res.faceLandmarks[0];
    } catch (e) { return; }

    if (!faceLm) return;

    const data = this.ctx.getImageData(0, 0, w, h);
    const filtered = new ImageData(new Uint8ClampedArray(data.data), w, h);
    this.applyFilter(filtered, name, this.intensity);
    this._tmpX.putImageData(filtered, 0, 0);

    const pts = faceLm.map(lm => [lm.x * w, lm.y * h]);

    this.ctx.save();
    this.ctx.beginPath();
    if (pts.length > 0) {
      this.ctx.moveTo(pts[0][0], pts[0][1]);
      // Convex hull approximation: use face oval landmarks
      const hullPts = this.convexHull(pts);
      for (const p of hullPts) this.ctx.lineTo(p[0], p[1]);
    }
    this.ctx.closePath();
    this.ctx.clip();
    this.ctx.drawImage(this._tmpC, 0, 0);
    this.ctx.restore();
  }

  convexHull(points) {
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (pts.length <= 3) return pts;
    const cross = (O, A, B) => (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0]);
    const lower = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
        lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
      while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], pts[i]) <= 0)
        upper.pop();
      upper.push(pts[i]);
    }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }

  renderDraw() {
    if (this.drawPoints.length < 3) return;
    const w = this.canvas.width, h = this.canvas.height;
    const name = FILTERS[this.filter];
    if (name === "None") return;

    const data = this.ctx.getImageData(0, 0, w, h);
    const filtered = new ImageData(new Uint8ClampedArray(data.data), w, h);
    this.applyFilter(filtered, name, this.intensity);
    this._tmpX.putImageData(filtered, 0, 0);

    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.moveTo(this.drawPoints[0][0], this.drawPoints[0][1]);
    for (let i = 1; i < this.drawPoints.length; i++)
      this.ctx.lineTo(this.drawPoints[i][0], this.drawPoints[i][1]);
    this.ctx.closePath();
    this.ctx.clip();
    this.ctx.drawImage(this._tmpC, 0, 0);
    this.ctx.restore();

    // Outline
    this.ctx.strokeStyle = "#00ff00";
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(this.drawPoints[0][0], this.drawPoints[0][1]);
    for (const p of this.drawPoints) this.ctx.lineTo(p[0], p[1]);
    this.ctx.closePath();
    this.ctx.stroke();
  }

  renderMagnifier(leftHand, rightHand) {
    if (!leftHand || !rightHand) return;
    const w = this.canvas.width, h = this.canvas.height;

    const tl = lmPx(leftHand[4], w, h);
    const tr = lmPx(rightHand[4], w, h);
    const br = lmPx(rightHand[8], w, h);
    const bl = lmPx(leftHand[8], w, h);

    const xs = [tl[0],tr[0],br[0],bl[0]], ys = [tl[1],tr[1],br[1],bl[1]];
    const x1 = Math.max(0, Math.min(...xs)), y1 = Math.max(0, Math.min(...ys));
    const x2 = Math.min(w, Math.max(...xs)), y2 = Math.min(h, Math.max(...ys));
    const bw = x2 - x1, bh = y2 - y1;
    if (bw < 10 || bh < 10) return;

    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const rw = bw / 2, rh = bh / 2;
    const rx = Math.max(0, cx - rw / 2), ry = Math.max(0, cy - rh / 2);
    const rw2 = Math.min(rw, w - rx), rh2 = Math.min(rh, h - ry);

    if (rw2 < 5 || rh2 < 5) return;

    const roi = this.ctx.getImageData(rx, ry, rw2, rh2);
    this._tmpX.clearRect(0, 0, w, h);
    this._tmpX.putImageData(roi, 0, 0);

    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.moveTo(...tl); this.ctx.lineTo(...tr);
    this.ctx.lineTo(...br); this.ctx.lineTo(...bl);
    this.ctx.closePath();
    this.ctx.clip();
    this.ctx.drawImage(this._tmpC, 0, 0, rw2, rh2, x1, y1, bw, bh);
    this.ctx.restore();

    this.ctx.strokeStyle = "#00ff00";
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(...tl); this.ctx.lineTo(...tr);
    this.ctx.lineTo(...br); this.ctx.lineTo(...bl);
    this.ctx.closePath();
    this.ctx.stroke();
  }

  // ── Main Loop ────────────────────────────────────────────────

  processFrame() {
    const now = performance.now();
    this.fpsTimes.push(now);
    while (this.fpsTimes.length > 0 && now - this.fpsTimes[0] > 1000) this.fpsTimes.shift();
    this.fps = this.fpsTimes.length;
    document.getElementById("fps-display").textContent = "FPS: " + this.fps;

    if (this.frozen && this.frozenData) {
      this.ctx.putImageData(this.frozenData, 0, 0);
      requestAnimationFrame(() => this.processFrame());
      return;
    }

    this.drawMirroredFrame();

    let ts = now;
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;

    let handResults = { landmarks: [], handedness: [] };
    try {
      handResults = this.handLandmarker.detectForVideo(this.canvas, ts);
    } catch (e) { /* skip frame */ }

    const { leftHand, rightHand } = this.processGestures(handResults);

    try {
      switch (this.mode) {
        case "NORMAL":    this.renderNormal(leftHand, rightHand); break;
        case "BACKGROUND":this.renderBackground(ts); break;
        case "FACE":      this.renderFace(ts); break;
        case "DRAW":      this.renderDraw(); break;
        case "MAGNIFIER": this.renderMagnifier(leftHand, rightHand); break;
      }
    } catch (e) { /* graceful degradation */ }

    this.drawHands(handResults);

    requestAnimationFrame(() => this.processFrame());
  }
}

// ── Launch ─────────────────────────────────────────────────────

const app = new GestureVision();
app.init();
