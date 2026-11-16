import os
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # idle inference threads sleep instead of busy-waiting (which shows up as high CPU)
os.environ.setdefault("KMP_BLOCKTIME", "0")
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np, torch, joblib, threading, time, os, sys, shutil, subprocess, webbrowser, psutil, json, re, runpy
from datetime import datetime
from collections import deque
from ultralytics import YOLO

torch.set_num_threads(min(2, os.cpu_count() or 1))  # 224px nano models barely speed up past 2 threads
cv2.setNumThreads(max(os.cpu_count() // 4, 1))  # OpenCV defaults to every core, which starves the UI on small CPUs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from modules.pose_utils import normalize_and_motion, extract_summary_features_v2
from modules.scene_utils import load_scene_model, predict_scene_frame
from modules.alert_utils import speak_alert, send_email_alert, send_sms_alert, send_telegram_message, telegram_find_chats
from modules.db_utils import init_db, get_recent_incidents, get_incidents, count_today_incidents, mark_false_alarm
from modules.pipeline import risk_from, save_evidence
import config
init_db()

# ---------- theme ----------
BG, PANEL, PANEL2, BORDER = "#0a0d16", "#10152a", "#151b33", "#1e2748"
TEXT, MUTED, ACCENT = "#e8ebf5", "#8a93ad", "#2f6bff"
RED, ORANGE, YELLOW, GREEN = "#ff4d5e", "#ff9a4d", "#ffcf4d", "#3ddc84"
LEVEL_COLOR = {"Critical": RED, "High": ORANGE, "Medium": YELLOW, "Low": GREEN}
LEVEL_BGR = {"Critical": (94, 77, 255), "High": (77, 154, 255), "Medium": (77, 207, 255), "Low": (132, 220, 61)}

ctk.set_appearance_mode("dark")

# ---------- models ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.path.join(ROOT, "models")
behavior_model = joblib.load(os.path.join(MODELS_DIR, "behavior_model.pkl"))
scene_model = load_scene_model(os.path.join(MODELS_DIR, "scene_model.pth"), device)
pose_model = YOLO(os.path.join(MODELS_DIR, "yolov8n-pose.pt"))

# ---------- tuning knobs ----------
SEQ_LEN, FRAME_SKIP, SCENE_SKIP = 30, 5, 15
PRE_SECONDS, POST_SECONDS = 5, 10
SCENE_WINDOW, SCENE_WARMUP, SCENE_RISE = 180, 20, 0.45  # scene alerts compare against the last 3 min of this camera; first 20 s are calibration
MAX_PEOPLE, MATCH_DIST, TRACK_TIMEOUT = 6, 0.2, 6  # people tracked per camera; max jump (fraction of the frame) between two checks; checks before a lost person is dropped
MIN_CHECKS = 4                          # risk must be re-checked at least this many times while High/Critical before it counts
RECONNECT_DELAY, GRID_SLOTS = 5, 4
SMALL_W, SMALL_H = 224, 168             # size fed to the models
EVIDENCE_WIDTH = 640                    # evidence clips are stored at most this wide (keeps RAM low on 1080p CCTV streams)
EVIDENCE_BLUR = os.path.join(ROOT, "evidence", "blurred")
CONFIG_PATH = os.path.join(ROOT, "config.py")
EVIDENCE_EVERY = 2                      # keep every Nth frame for evidence (2 = clips are saved at half the camera fps: half the RAM, CPU and save time)
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 75]
PAGE_LIMIT = 40                         # rows shown on the Alerts / Evidence pages (building hundreds of widgets freezes the window)
VISIBLE = {"dash": True}                # camera threads skip drawing preview frames while another page is open
os.makedirs(EVIDENCE_BLUR, exist_ok=True)

# adjustable from the Settings page, saved to settings.json so they survive a restart
SETTINGS_PATH = os.path.join(ROOT, "settings.json")
SETTINGS = {"confirm": 4.0, "scene_thresh": 0.85, "voice": True, "scene_onset": True, "multi_person": True}
try:
    with open(SETTINGS_PATH) as f:
        SETTINGS.update(json.load(f))
except Exception:
    pass


def save_settings():
    with open(SETTINGS_PATH, "w") as f:
        json.dump(SETTINGS, f, indent=2)


def open_path(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def open_in_editor(path):
    if sys.platform.startswith("win"):
        subprocess.Popen(["notepad", path])  # os.startfile on a .py file could run it instead of opening it
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-t", path])
    else:
        subprocess.Popen(["xdg-open", path])


def save_copy(path):
    dest = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=os.path.basename(path))
    if dest:
        shutil.copy(path, dest)


def level_of(score):
    return "Critical" if score >= 80 else "High" if score >= 60 else "Medium" if score >= 35 else "Low"


class PersonTracker:
    """Follows every person across the sampled frames, so each one gets their own movement history."""

    def __init__(self):
        self.tracks, self.next_id, self.cycle = {}, 1, 0

    def update(self, keypoints, boxes, size, max_people=MAX_PEOPLE):
        """keypoints (n,17,2) normalized, boxes (n,4) in model-input pixels, size=(w, h). Returns {id: track}."""
        self.cycle += 1
        w, h = size
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        keep = sorted(range(len(boxes)), key=lambda i: -areas[i])[:max_people]  # biggest boxes = closest to the camera
        centers = {i: ((boxes[i][0] + boxes[i][2]) / 2 / w, (boxes[i][1] + boxes[i][3]) / 2 / h) for i in keep}
        pairs = sorted((np.hypot(centers[i][0] - t["center"][0], centers[i][1] - t["center"][1]), i, tid)
                       for i in keep for tid, t in self.tracks.items())
        assign, used = {}, set()
        for d, i, tid in pairs:  # closest pairs first, each person and each track used once
            if d > MATCH_DIST:
                break
            if i not in assign and tid not in used:
                assign[i] = tid
                used.add(tid)
        for i in keep:
            tid = assign.get(i)
            if tid is None:  # someone new walked in
                tid, self.next_id = self.next_id, self.next_id + 1
                self.tracks[tid] = {"buffer": deque(maxlen=SEQ_LEN), "prob": 0.0}
            t = self.tracks[tid]
            t["center"], t["box"], t["seen"] = centers[i], boxes[i], self.cycle
            t["buffer"].append(keypoints[i].flatten())
        for tid in [k for k, t in self.tracks.items() if self.cycle - t["seen"] > TRACK_TIMEOUT]:
            del self.tracks[tid]  # person left the picture
        return self.tracks


def scene_gate(raw, now, hist, latched, thr, onset_on):
    """A high scene-model score only counts if it just JUMPED above this camera's own recent normal.
    An empty room that scores high all the time never jumps, so it is ignored; a fire or explosion
    appears suddenly, so it passes. Returns (effective score, latched)."""
    hist.append((now, raw))
    while now - hist[0][0] > SCENE_WINDOW:
        hist.popleft()
    if raw < thr:
        latched = False
    elif not onset_on:
        latched = True
    elif not latched:
        old = [r for t, r in hist if now - t > 3]  # recent seconds excluded, so the jump itself can't hide in the baseline
        if now - hist[0][0] >= SCENE_WARMUP and len(old) >= 5 and raw - float(np.median(old)) >= SCENE_RISE:
            latched = True
    return (raw if (raw >= thr and latched) else raw * 0.3), latched


LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def sms_ready():
    return bool(getattr(config, "SMS_ENABLED", False) and getattr(config, "TWILIO_SID", "") and getattr(config, "TWILIO_TOKEN", "")
                and getattr(config, "TWILIO_FROM", "") and getattr(config, "SMS_RECIPIENTS", []))


def sms_status_text():
    if not getattr(config, "SMS_ENABLED", False):
        return "SMS alerts: off. Set SMS_ENABLED = True and the Twilio details in config.py to turn them on."
    if not sms_ready():
        return "SMS alerts: enabled but incomplete. Fill in TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM and SMS_RECIPIENTS in config.py."
    return (f"SMS alerts: on for {len(config.SMS_RECIPIENTS)} number(s). Sent for {getattr(config, 'SMS_MIN_LEVEL', 'Critical')} incidents, "
            f"at most one per camera every {getattr(config, 'SMS_COOLDOWN', 120)} s.")


def tg_ready():
    return bool(getattr(config, "TELEGRAM_ENABLED", False) and getattr(config, "TELEGRAM_BOT_TOKEN", "") and getattr(config, "TELEGRAM_CHAT_IDS", []))


def tg_status_text():
    if not getattr(config, "TELEGRAM_ENABLED", False):
        return "Telegram alerts: off (free). Set TELEGRAM_ENABLED = True, your bot token and chat ids in config.py. See the README."
    if not tg_ready():
        return "Telegram alerts: enabled but incomplete. Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS (the Find chat IDs button helps)."
    clip = "with the blurred clip" if getattr(config, "TELEGRAM_SEND_CLIP", False) else "text only"
    return (f"Telegram alerts: on for {len(config.TELEGRAM_CHAT_IDS)} chat(s), {getattr(config, 'TELEGRAM_MIN_LEVEL', 'High')} and above, {clip}, "
            f"at most one per camera every {getattr(config, 'TELEGRAM_COOLDOWN', 60)} s.")


def send_incident_telegram(level, score, category, cam_id, place):
    what = "suspicious behavior" if category == "Behavior" else "scene anomaly"
    text = f"🚨 SafeVision ALERT: {level} ({score:.0f}/100) {what} at Camera {cam_id} ({place.replace('_', ' ').title()}) at {datetime.now():%H:%M:%S}. Open the dashboard to review."
    delivered, errors = send_telegram_message(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_IDS, text)
    print(f"[telegram] delivered to {delivered} of {len(config.TELEGRAM_CHAT_IDS)} chat(s)" + (f" | errors: {errors}" if errors else ""))


def send_incident_sms(level, score, category, cam_id, place):
    what = "suspicious behavior" if category == "Behavior" else "scene anomaly"  # plain ASCII keeps it to one 160-character SMS
    body = f"SafeVision ALERT: {level} ({score:.0f}/100) {what} at Camera {cam_id} ({place.replace('_', ' ').title()}) {datetime.now():%H:%M}. Check the dashboard."
    accepted, errors = send_sms_alert(config.TWILIO_SID, config.TWILIO_TOKEN, config.TWILIO_FROM, config.SMS_RECIPIENTS, body)
    print(f"[sms] accepted for {accepted} of {len(config.SMS_RECIPIENTS)} number(s)" + (f" | errors: {errors}" if errors else ""))


def tile_title(cam, slot):
    return (f"Camera {cam['id']} - {cam['location'].replace('_', ' ').title()} ({cam_type(cam['source'])})"
            if cam else f"Camera {slot} - Not Available")


def put(lbl, text, **kw):
    if lbl.cget("text") != text:  # CTk labels redraw on every configure(), so only touch them when the text really changed
        lbl.configure(text=text, **kw)


def cam_type(source):
    return "Webcam" if isinstance(source, int) else "CCTV" if str(source).startswith("rtsp") else "IP Camera"


def current_time_of_day():
    return "night" if datetime.now().hour < 6 or datetime.now().hour >= 20 else "day"


def auto_detect_cameras(max_check=5, skip=()):
    found = []
    for i in range(max_check):
        if i in skip:
            continue  # already open in a running tile, so a second open would fail
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({"id": i + 1, "source": i, "location": "general", "time_of_day": "auto"})
        cap.release()
    return found


def evidence_size():
    total = 0
    for root, _, files in os.walk(os.path.join(ROOT, "evidence")):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return f"{total / 1e9:.2f} GB" if total >= 1e9 else f"{total / 1e6:.0f} MB"


def fmt_ts(ts, fmt):
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime(fmt)
    except Exception:
        return ts


def draw_badge(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(y - th - 8, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 10, y0 + th + 8), color, -1)
    cv2.putText(img, text, (x + 5, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ---------- camera tile (one thread per camera) ----------
class CameraTile(ctk.CTkFrame):
    """One camera. The worker thread only computes; it never touches Tk. The main thread polls `latest` ~10x/sec."""

    def __init__(self, master, cam, slot, dw, dh):
        super().__init__(master, corner_radius=12, fg_color=PANEL, border_width=1, border_color=BORDER)
        self.cam, self.dw, self.dh = cam, dw, dh
        self.level, self.score, self.fps, self.online, self.msg = "Low", 0.0, 0.0, False, "Connecting..."
        self.saving, self.pre_frames, self.post_frames, self.save_meta = False, [], [], {}
        self.last_sms = self.last_tg = 0.0
        self.running, self.thread = bool(cam), None
        self.latest, self._shown, self._has_text, self._cache, self._ui_t = None, None, True, {}, 0.0

        title = tile_title(cam, slot)
        self.slot, self._poll_id = slot, None
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(8, 4))
        self.title_lbl = ctk.CTkLabel(top, text=title, font=("Segoe UI", 12, "bold"), text_color=TEXT)
        self.title_lbl.pack(side="left")
        self.state_lbl = ctk.CTkLabel(top, text="● Connecting" if cam else "● Offline", font=("Segoe UI", 11), text_color=MUTED)
        self.state_lbl.pack(side="right")

        # one plain tk.Label + one reusable PhotoImage: pixels are updated in place instead of allocating a new image every frame
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (dw, dh), (7, 9, 16)))
        self.video_lbl = tk.Label(self, image=self.photo, text="Connecting..." if cam else "Not Available", compound="center",
                                  fg=MUTED, bg="#070910", bd=0, highlightthickness=0, font=("Segoe UI", 11))
        self.video_lbl.pack(padx=12, pady=(0, 4))

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=12, pady=(0, 8))
        self.mode_lbl = ctk.CTkLabel(bottom, text="", font=("Segoe UI", 11), text_color=MUTED)
        self.mode_lbl.pack(side="left")
        self.time_lbl = ctk.CTkLabel(bottom, text="", font=("Segoe UI", 11), text_color=MUTED)
        self.time_lbl.pack(side="right")
        self.risk_lbl = ctk.CTkLabel(bottom, text="", font=("Segoe UI", 11, "bold"))
        self.risk_lbl.pack(side="right", padx=12)

        if cam:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
            self.poll()

    def upd(self, key, widget, **kw):
        if self._cache.get(key) != kw:
            widget.configure(**kw)
            self._cache[key] = kw

    def poll(self):
        if not self.running:
            return
        rgb = self.latest
        if rgb is not None and rgb is not self._shown:
            self._shown = rgb
            self.photo.paste(Image.fromarray(rgb))
            if self._has_text:
                self.video_lbl.configure(text="")
                self._cache.pop("msg", None)
                self._has_text = False
        elif self.msg and not self.online:
            self.upd("msg", self.video_lbl, text=self.msg)
            self._has_text = True

        now = time.time()
        if now - self._ui_t >= 0.5:  # border and labels only need ~2 refreshes per second
            self._ui_t = now
            level = self.level
            hot = level in ("High", "Critical")
            self.upd("border", self, border_color=LEVEL_COLOR[level] if hot else BORDER, border_width=2 if hot else 1)
            state = "● Online" if self.online else ("● Connecting" if self.msg == "Connecting..." else "● Retrying")
            self.upd("state", self.state_lbl, text=state, text_color=GREEN if self.online else ORANGE)
            self.upd("risk", self.risk_lbl, text=f"{level} · {self.score:.0f}" if self.online else "", text_color=LEVEL_COLOR[level])
            self.upd("mode", self.mode_lbl, text="● Recording evidence" if self.saving else "● Monitoring", text_color=RED if self.saving else MUTED)
        self._poll_id = self.after(100, self.poll)

    def loop(self):
        cam_id = self.cam["id"]
        tracker = PersonTracker()
        sx, sy = self.dw / SMALL_W, self.dh / SMALL_H

        while self.running:
            cap = cv2.VideoCapture(self.cam["source"])
            if not cap.isOpened():
                self.online, self.msg = False, "Unreachable - retrying..."
                time.sleep(RECONNECT_DELAY)
                continue

            nominal = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_buffer = deque(maxlen=max(int(PRE_SECONDS * nominal / EVIDENCE_EVERY), 1))  # holds small JPEGs, not raw frames
            behavior_prob = scene_prob = 0.0
            tracks = tracker.tracks
            scene_hist, scene_latched = deque(), False
            count, high_since, high_checks, mark, event_active = 0, None, 0, time.time(), False
            self.online, self.msg = True, ""

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                small = cv2.resize(frame, (SMALL_W, SMALL_H))
                checked = False

                if count % FRAME_SKIP == 0:
                    result = pose_model(small, verbose=False, imgsz=224)[0]
                    if result.keypoints is not None and len(result.keypoints.xyn) > 0:
                        kps, boxes = result.keypoints.xyn.cpu().numpy(), result.boxes.xyxy.cpu().numpy()
                    else:
                        kps, boxes = np.zeros((0, 17, 2)), np.zeros((0, 4))
                    tracks = tracker.update(kps, boxes, (SMALL_W, SMALL_H), MAX_PEOPLE if SETTINGS["multi_person"] else 1)
                    ready = [t for t in tracks.values() if len(t["buffer"]) == SEQ_LEN]
                    if ready:  # every person gets their own prediction from their own movement
                        feats = np.vstack([extract_summary_features_v2(normalize_and_motion(np.array(t["buffer"]))) for t in ready])
                        for t, p in zip(ready, behavior_model.predict_proba(feats)[:, 1]):
                            t["prob"] = float(p)
                    behavior_prob = max((t["prob"] for t in tracks.values()), default=0.0)  # the alert follows the most suspicious person
                    checked = True
                if count % SCENE_SKIP == 0:
                    scene_prob, scene_latched = scene_gate(predict_scene_frame(scene_model, device, small), time.time(), scene_hist,
                                                           scene_latched, SETTINGS["scene_thresh"], SETTINGS["scene_onset"])
                    checked = True

                tod = current_time_of_day() if self.cam["time_of_day"] == "auto" else self.cam["time_of_day"]
                loc_mult = 1.3 if self.cam["location"] in ("atm", "gate", "server_room") else 1.0  # read live, so edits apply without a reconnect
                time_mult = 1.2 if tod == "night" else 1.0
                score = min(risk_from(max(behavior_prob, scene_prob)) * loc_mult * time_mult, 100)
                level = level_of(score)
                category = "Behavior" if behavior_prob >= scene_prob else "Scene"
                self.level, self.score = level, score

                # evidence frames are stored as small JPEGs (~20-50 KB) instead of raw arrays (~0.9 MB): same clip, ~20x less RAM
                store, buf = count % EVIDENCE_EVERY == 0, None
                if store:
                    ev = frame if frame.shape[1] <= EVIDENCE_WIDTH else cv2.resize(
                        frame, (EVIDENCE_WIDTH, (frame.shape[0] * EVIDENCE_WIDTH // frame.shape[1]) // 2 * 2))
                    enc_ok, buf = cv2.imencode(".jpg", ev, JPEG_PARAMS)
                    if enc_ok:
                        frame_buffer.append(buf)
                    else:
                        store = False

                # only a risk that stays High/Critical (and keeps being re-checked) counts as a real incident
                confirmed_now = level in ("High", "Critical")
                if confirmed_now and high_since is None:
                    high_since, high_checks = time.time(), 0
                elif not confirmed_now:
                    high_since, high_checks = None, 0
                if confirmed_now and checked:
                    high_checks += 1
                sustained = confirmed_now and high_since and (time.time() - high_since) >= SETTINGS["confirm"] and high_checks >= MIN_CHECKS
                clip_fps = max((self.fps or nominal) / EVIDENCE_EVERY, 1.0)  # playback rate of the stored clip = real time

                if sustained and not self.saving and not event_active:
                    event_active, self.saving = True, True
                    self.post_frames, self.pre_frames = [], list(frame_buffer)
                    self.save_meta = {"level": level, "risk_score": round(score, 1), "category": category,
                                      "behavior_conf": round(float(behavior_prob), 2), "scene_conf": round(float(scene_prob), 2),
                                      "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "camera_id": cam_id}
                    if SETTINGS["voice"]:
                        threading.Thread(target=speak_alert, args=(f"{level} alert on camera {cam_id}. {category} anomaly detected.",), daemon=True).start()
                    min_rank = LEVEL_RANK.get(getattr(config, "SMS_MIN_LEVEL", "Critical"), 3)  # SMS can cost money, so only serious incidents by default
                    if sms_ready() and LEVEL_RANK[level] >= min_rank and time.time() - self.last_sms >= getattr(config, "SMS_COOLDOWN", 120):
                        self.last_sms = time.time()
                        threading.Thread(target=send_incident_sms, args=(level, score, category, cam_id, self.cam["location"])).start()  # non-daemon: the text still goes out if the window closes
                    tg_rank = LEVEL_RANK.get(getattr(config, "TELEGRAM_MIN_LEVEL", "High"), 2)  # free, so it can alert earlier than SMS
                    if tg_ready() and LEVEL_RANK[level] >= tg_rank and time.time() - self.last_tg >= getattr(config, "TELEGRAM_COOLDOWN", 60):
                        self.last_tg = time.time()
                        threading.Thread(target=send_incident_telegram, args=(level, score, category, cam_id, self.cam["location"])).start()
                if not confirmed_now:
                    event_active = False
                if self.saving and store:
                    self.post_frames.append(buf)
                    if len(self.post_frames) >= int(POST_SECONDS * clip_fps):
                        clip, meta = self.pre_frames + self.post_frames, dict(self.save_meta)
                        self.saving, self.pre_frames, self.post_frames = False, [], []
                        threading.Thread(target=save_evidence, args=(clip, meta, clip_fps)).start()  # non-daemon: finishes even if you close the window

                count += 1
                if count % 30 == 0:
                    now = time.time()
                    self.fps, mark = 30 / (now - mark), now

                if VISIBLE["dash"] and count % 3 == 0:  # ~10 preview frames/sec, and none while another page is open
                    disp = cv2.resize(frame, (self.dw, self.dh))
                    color = LEVEL_BGR[level]
                    hot = level in ("High", "Critical")
                    shown = 0
                    for tid, t in tracks.items():
                        if tracker.cycle - t["seen"] > 1:
                            continue  # not seen in the latest check, so no box
                        p_level = level_of(min(risk_from(t["prob"]) * loc_mult * time_mult, 100))  # each person is judged on their own
                        label = ("Suspicious" if p_level in ("High", "Critical") else "Watch" if p_level == "Medium" else "Normal") + f" #{tid}"
                        x1, y1, x2, y2 = (int(v) for v in (t["box"][0] * sx, t["box"][1] * sy, t["box"][2] * sx, t["box"][3] * sy))
                        cv2.rectangle(disp, (x1, y1), (x2, y2), LEVEL_BGR[p_level], 2)
                        draw_badge(disp, label, x1, y1, LEVEL_BGR[p_level])
                        shown += 1
                    if hot and (category == "Scene" or not shown):
                        draw_badge(disp, "Anomaly" if category == "Scene" else "Suspicious", 8, 30, color)
                    self.latest = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            cap.release()
            self.level = "Low"
            if self.running:
                self.online, self.msg = False, "Stream dropped - reconnecting..."
                time.sleep(RECONNECT_DELAY)

    def update_cam(self, cam):
        """Same camera, edited settings (location / time of day): apply without reconnecting."""
        self.cam = cam
        self.title_lbl.configure(text=tile_title(cam, self.slot))

    def stop_async(self):
        """Stop this tile without blocking the window; a half-captured incident is still saved in the background."""
        self.running = False
        if self._poll_id:
            try:
                self.after_cancel(self._poll_id)
            except Exception:
                pass

        def finish():
            if self.thread:
                self.thread.join(timeout=3)
            if self.saving and self.post_frames:
                save_evidence(self.pre_frames + self.post_frames, dict(self.save_meta), max((self.fps or 25) / EVIDENCE_EVERY, 1.0))
                self.saving = False

        threading.Thread(target=finish).start()  # non-daemon: finishes even if the app closes meanwhile

    def shutdown(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.saving and self.post_frames:  # window closed mid-incident: save what was captured instead of losing it
            print(f"[Cam {self.cam['id']}] saving partial evidence before exit...")
            save_evidence(self.pre_frames + self.post_frames, dict(self.save_meta), max((self.fps or 25) / EVIDENCE_EVERY, 1.0))
            self.saving = False


# ---------- small widgets ----------
class StatCard(ctk.CTkFrame):
    def __init__(self, master, icon, title, color):
        super().__init__(master, corner_radius=12, fg_color=PANEL, border_width=1, border_color=BORDER)
        ctk.CTkLabel(self, text=icon, font=("Segoe UI Emoji", 20), width=46, height=46, corner_radius=10,
                     fg_color=PANEL2, text_color=color).pack(side="left", padx=12, pady=12)
        box = ctk.CTkFrame(self, fg_color="transparent")
        box.pack(side="left", fill="both", expand=True, pady=8)
        self.title = ctk.CTkLabel(box, text=title, font=("Segoe UI", 11), text_color=MUTED, anchor="w")
        self.title.pack(anchor="w")
        self.value = ctk.CTkLabel(box, text="-", font=("Segoe UI", 22, "bold"), text_color=TEXT, anchor="w")
        self.value.pack(anchor="w")
        self.sub = ctk.CTkLabel(box, text="", font=("Segoe UI", 11), text_color=MUTED, anchor="w")
        self.sub.pack(anchor="w")

    def set(self, value, sub="", sub_color=MUTED, title=None):
        key = (value, sub, sub_color, title)
        if getattr(self, "_last", None) == key:
            return
        self._last = key
        self.value.configure(text=value)
        self.sub.configure(text=sub, text_color=sub_color)
        if title:
            self.title.configure(text=title)


# ---------- window ----------
app = ctk.CTk()
app.title("SafeVision AI - Real-Time Threat Detection")
sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
app.geometry(f"{min(1500, sw - 40)}x{min(900, sh - 80)}+10+10")
app.minsize(1100, 650)
app.configure(fg_color=BG)
scale = max(0.7, min(1.0, (sh - 80) / 900))   # shrink camera tiles on small screens
DW, DH = int(380 * scale), int(235 * scale)

cameras = config.CAMERAS or auto_detect_cameras()
slots = cameras[:GRID_SLOTS] + [None] * max(0, GRID_SLOTS - len(cameras))
cam_by_id = {str(c["id"]): c for c in cameras}

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# ----- sidebar -----
sidebar = ctk.CTkFrame(app, width=210, corner_radius=0, fg_color=PANEL)
sidebar.grid(row=0, column=0, sticky="ns")
sidebar.grid_propagate(False)
ctk.CTkLabel(sidebar, text="🛡  SafeVision AI", font=("Segoe UI", 18, "bold"), text_color=TEXT).pack(padx=16, pady=(20, 18), anchor="w")


nav_buttons = {}


def nav_btn(name, text, icon):
    b = ctk.CTkButton(sidebar, text=f"  {icon}   {text}", anchor="w", height=40, corner_radius=8, font=("Segoe UI", 13),
                      fg_color="transparent", hover_color=PANEL2, text_color=TEXT, command=lambda: show_page(name))
    b.pack(fill="x", padx=10, pady=2)
    nav_buttons[name] = b


nav_btn("dashboard", "Dashboard", "🏠")
nav_btn("alerts", "Alerts", "🔔")
nav_btn("evidence", "Evidence Library", "📁")
nav_btn("settings", "Settings", "⚙")

status_panel = ctk.CTkFrame(sidebar, corner_radius=12, fg_color=PANEL2, border_width=1, border_color=BORDER)
status_panel.pack(side="bottom", fill="x", padx=10, pady=12)
sys_state = ctk.CTkLabel(status_panel, text="● Starting...", font=("Segoe UI", 12, "bold"), text_color=MUTED, anchor="w")
sys_state.pack(fill="x", padx=12, pady=(10, 4))
sys_vals = {}
for key in ("Cameras Online", "Processing", "PC CPU", "App CPU", "Memory Usage", "App RAM"):
    r = ctk.CTkFrame(status_panel, fg_color="transparent")
    r.pack(fill="x", padx=12, pady=1)
    ctk.CTkLabel(r, text=key, font=("Segoe UI", 11), text_color=MUTED).pack(side="left")
    sys_vals[key] = ctk.CTkLabel(r, text="-", font=("Segoe UI", 11, "bold"), text_color=TEXT)
    sys_vals[key].pack(side="right")
ctk.CTkFrame(status_panel, height=6, fg_color="transparent").pack()

# ----- main area -----
main = ctk.CTkFrame(app, fg_color="transparent")
main.grid(row=0, column=1, sticky="nsew", padx=14, pady=10)
main.grid_columnconfigure(0, weight=1)
main.grid_rowconfigure(1, weight=1)

header = ctk.CTkFrame(main, fg_color="transparent")
header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
ctk.CTkLabel(header, text="SafeVision AI", font=("Segoe UI", 22, "bold"), text_color=TEXT).pack(side="left")
ctk.CTkLabel(header, text="   Real-Time Threat Detection & Alert System", font=("Segoe UI", 12), text_color=MUTED).pack(side="left")
ctk.CTkLabel(header, text="👤 Operator", font=("Segoe UI", 12), text_color=TEXT).pack(side="right", padx=(14, 0))
bell = ctk.CTkLabel(header, text="🔔", font=("Segoe UI Emoji", 14), text_color=TEXT)
bell.pack(side="right", padx=14)
ctk.CTkLabel(header, text="● System Online", font=("Segoe UI", 12), text_color=GREEN).pack(side="right")

page_host = ctk.CTkFrame(main, fg_color="transparent")
page_host.grid(row=1, column=0, sticky="nsew")
page_host.grid_columnconfigure(0, weight=1)
page_host.grid_rowconfigure(0, weight=1)
pages = {}
dash_page = ctk.CTkFrame(page_host, fg_color="transparent")
dash_page.grid(row=0, column=0, sticky="nsew")
dash_page.grid_columnconfigure(0, weight=1)
dash_page.grid_rowconfigure(1, weight=1)
pages["dashboard"] = dash_page

cards_row = ctk.CTkFrame(dash_page, fg_color="transparent")
cards_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
cards = {k: StatCard(cards_row, icon, title, color) for k, (icon, title, color) in {
    "cams": ("🎥", "Total Cameras", ACCENT), "alerts": ("🔔", "Active Alerts", RED),
    "threats": ("🛡", "Threats Detected (Today)", GREEN), "evidence": ("💾", "Evidence Stored", "#8b6cff"),
    "time": ("📅", "", ACCENT)}.items()}
for i, c in enumerate(cards.values()):
    cards_row.grid_columnconfigure(i, weight=1, uniform="cards")
    c.grid(row=0, column=i, sticky="ew", padx=4)

content = ctk.CTkFrame(dash_page, fg_color="transparent")
content.grid(row=1, column=0, sticky="nsew")
content.grid_columnconfigure(0, weight=1)
content.grid_columnconfigure(1, weight=0)
content.grid_rowconfigure(0, weight=1)

# camera grid
cam_panel = ctk.CTkFrame(content, corner_radius=14, fg_color=PANEL, border_width=1, border_color=BORDER)
cam_panel.grid(row=0, column=0, sticky="nsew")
cam_head = ctk.CTkFrame(cam_panel, fg_color="transparent")
cam_head.pack(fill="x", padx=16, pady=(12, 6))
ctk.CTkLabel(cam_head, text="●  Live Camera Feeds", font=("Segoe UI", 15, "bold"), text_color=TEXT).pack(side="left")
cam_note = ctk.CTkLabel(cam_head, text="", font=("Segoe UI", 11), text_color=MUTED)
cam_note.pack(side="right")
grid = ctk.CTkScrollableFrame(cam_panel, fg_color="transparent")  # scrolls when the tiles are taller than the window
grid.pack(fill="both", expand=True, padx=8, pady=(0, 8))
tiles = [CameraTile(grid, cam, i + 1, DW, DH) for i, cam in enumerate(slots)]
for i, t in enumerate(tiles):
    grid.grid_columnconfigure(i % 2, weight=1)
    t.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="nsew")

# right column: recent alerts + selected alert details
right = ctk.CTkFrame(content, fg_color="transparent", width=350)
right.grid(row=0, column=1, sticky="ns", padx=(10, 0))
right.grid_propagate(False)
right.grid_rowconfigure(0, weight=1)

alerts_panel = ctk.CTkFrame(right, corner_radius=14, fg_color=PANEL, border_width=1, border_color=BORDER)
alerts_panel.grid(row=0, column=0, sticky="nsew")
ah = ctk.CTkFrame(alerts_panel, fg_color="transparent")
ah.pack(fill="x", padx=14, pady=(12, 4))
ctk.CTkLabel(ah, text="🔔  Recent Alerts", font=("Segoe UI", 15, "bold"), text_color=TEXT).pack(side="left")
ctk.CTkButton(ah, text="View All", width=60, height=24, fg_color="transparent", hover_color=PANEL2, text_color=ACCENT,
              command=lambda: show_page("alerts")).pack(side="right")
alerts_list = ctk.CTkScrollableFrame(alerts_panel, fg_color="transparent")
alerts_list.pack(fill="both", expand=True, padx=6, pady=(0, 8))

detail = ctk.CTkFrame(right, corner_radius=14, fg_color=PANEL, border_width=1, border_color=BORDER)
detail.grid(row=1, column=0, sticky="ew", pady=(10, 0))
dh = ctk.CTkFrame(detail, fg_color="transparent")
dh.pack(fill="x", padx=14, pady=(12, 6))
ctk.CTkLabel(dh, text="Selected Alert Details", font=("Segoe UI", 14, "bold"), text_color=TEXT).pack(side="left")
d_badge = ctk.CTkLabel(dh, text="", font=("Segoe UI", 10, "bold"))
d_badge.pack(side="right")
d_title = ctk.CTkLabel(detail, text="No alert selected", font=("Segoe UI", 12, "bold"), text_color=TEXT, anchor="w")
d_title.pack(fill="x", padx=14)
dmid = ctk.CTkFrame(detail, fg_color="transparent")
dmid.pack(fill="x", padx=14, pady=6)
BLANK = ctk.CTkImage(Image.new("RGB", (120, 84), "#070910"), size=(120, 84))
d_thumb = ctk.CTkLabel(dmid, text="No preview", image=BLANK, width=120, height=84, text_color=MUTED, corner_radius=8)
d_thumb.pack(side="left")
dinfo = ctk.CTkFrame(dmid, fg_color="transparent")
dinfo.pack(side="left", padx=10, fill="x", expand=True)
d_vals = {}
for key in ("Camera ID", "Location", "Timestamp", "Risk Score"):
    r = ctk.CTkFrame(dinfo, fg_color="transparent")
    r.pack(fill="x", pady=1)
    ctk.CTkLabel(r, text=key, font=("Segoe UI", 11), text_color=MUTED, width=70, anchor="w").pack(side="left")
    d_vals[key] = ctk.CTkLabel(r, text="-", font=("Segoe UI", 11, "bold"), text_color=TEXT, anchor="w")
    d_vals[key].pack(side="left")
d_reason = ctk.CTkLabel(detail, text="", font=("Segoe UI", 11), text_color=MUTED, wraplength=310, justify="left", anchor="w")
d_reason.pack(fill="x", padx=14, pady=(0, 6))
dbtn = ctk.CTkFrame(detail, fg_color="transparent")
dbtn.pack(fill="x", padx=14, pady=(0, 12))

state = {"sel": None, "ids": None, "tick": 0, "page": "dashboard", "alerts_top": None, "cfg_seen": None, "cfg_due": None}


def evidence_file(r):
    return os.path.join(ROOT, r[9])  # blurred copy only: privacy by default, raw footage stays admin-only


def view_video():
    r = state["sel"]
    if r and os.path.exists(evidence_file(r)):
        open_path(evidence_file(r))
    else:
        messagebox.showwarning("Evidence", "Select an alert whose evidence file exists first.")


def download_evidence():
    r = state["sel"]
    if not (r and os.path.exists(evidence_file(r))):
        return messagebox.showwarning("Evidence", "Select an alert whose evidence file exists first.")
    dest = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=os.path.basename(r[9]))
    if dest:
        shutil.copy(evidence_file(r), dest)


ctk.CTkButton(dbtn, text="▶  View Full Video", height=34, fg_color=ACCENT, command=view_video).pack(side="left", expand=True, fill="x", padx=(0, 4))
ctk.CTkButton(dbtn, text="⬇  Download", height=34, fg_color=PANEL2, border_width=1, border_color=BORDER, command=download_evidence).pack(side="left", expand=True, fill="x", padx=(4, 0))


def select_incident(r):
    state["sel"] = r
    color = LEVEL_COLOR.get(r[3], MUTED)
    cam = cam_by_id.get(str(r[1]))
    d_title.configure(text="Suspicious Behavior" if r[5] == "Behavior" else "Scene Anomaly")
    d_badge.configure(text=r[3].upper(), text_color=color)
    d_vals["Camera ID"].configure(text=f"CAM-{int(r[1]):02d}" if str(r[1]).isdigit() else str(r[1]))
    d_vals["Location"].configure(text=cam["location"].replace("_", " ").title() if cam else "-")
    d_vals["Timestamp"].configure(text=fmt_ts(r[2], "%d %b %H:%M:%S"))
    d_vals["Risk Score"].configure(text=f"{r[4]:.0f} / 100", text_color=color)
    d_reason.configure(text=(f"Sustained suspicious body motion detected (behavior confidence {r[6]:.2f})." if r[5] == "Behavior"
                             else f"Scene anomaly detected (scene confidence {r[7]:.2f})."))
    thumb = BLANK
    path = evidence_file(r)
    if os.path.exists(path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, f = cap.read()
        cap.release()
        if ok:
            thumb = ctk.CTkImage(Image.fromarray(cv2.cvtColor(cv2.resize(f, (120, 84)), cv2.COLOR_BGR2RGB)), size=(120, 84))
    d_thumb.configure(image=thumb, text="" if thumb is not BLANK else "No preview")
    d_thumb._keep = thumb


def render_alerts():
    rows = get_recent_incidents(8)
    ids = tuple(r[0] for r in rows)
    if ids == state["ids"]:
        return  # nothing new, don't rebuild (keeps scroll position)
    state["ids"] = ids
    for w in alerts_list.winfo_children():
        w.destroy()
    if not rows:
        ctk.CTkLabel(alerts_list, text="No incidents yet.\nAn incident is saved only after risk stays\nHigh/Critical for a few seconds.", text_color=MUTED, justify="center").pack(pady=20)
        return
    for r in rows:
        color = LEVEL_COLOR.get(r[3], MUTED)
        row = ctk.CTkFrame(alerts_list, fg_color=PANEL2, corner_radius=8, border_width=1, border_color=BORDER)
        row.pack(fill="x", pady=3, padx=2)
        bar = ctk.CTkFrame(row, width=4, fg_color=color, corner_radius=2)
        bar.pack(side="left", fill="y", padx=(0, 8), pady=6)
        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, pady=6)
        ctk.CTkLabel(info, text="Suspicious Behavior Detected" if r[5] == "Behavior" else "Scene Anomaly Detected",
                     font=("Segoe UI", 12, "bold"), text_color=TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(info, text=f"Camera {r[1]} · {fmt_ts(r[2], '%H:%M:%S')}", font=("Segoe UI", 11), text_color=MUTED, anchor="w").pack(anchor="w")
        ctk.CTkLabel(row, text=r[3].upper(), font=("Segoe UI", 10, "bold"), text_color=color).pack(side="right", padx=10)
        for w in (row, bar, info, *info.winfo_children(), *row.winfo_children()):
            w.bind("<Button-1>", lambda e, r=r: select_incident(r))
    if state["sel"] is None or state["sel"][0] not in ids:
        select_incident(rows[0])


# ---------- extra pages: Alerts, Evidence Library, Settings ----------
COLS = [("Time", 120), ("Camera", 70), ("Level", 75), ("Risk", 50), ("Source", 75), ("Behavior", 75), ("Scene", 60)]
EMPTY_MSG = ("No incidents yet.\nAn incident is saved only after risk stays High/Critical for a few seconds,\n"
             "so this stays empty until the system actually confirms a threat.")


def page_shell(title):
    page = ctk.CTkFrame(page_host, corner_radius=14, fg_color=PANEL, border_width=1, border_color=BORDER)
    page.grid(row=0, column=0, sticky="nsew")
    top = ctk.CTkFrame(page, fg_color="transparent")
    top.pack(fill="x", padx=16, pady=(14, 8))
    ctk.CTkLabel(top, text=title, font=("Segoe UI", 16, "bold"), text_color=TEXT).pack(side="left")
    return page, top


def small_btn(parent, text, command, width=90, primary=False):
    return ctk.CTkButton(parent, text=text, width=width, height=28, command=command,
                         fg_color=ACCENT if primary else PANEL2, border_width=0 if primary else 1, border_color=BORDER)


# ----- Alerts page -----
alerts_page, a_top = page_shell("🔔  All Alerts")
pages["alerts"] = alerts_page
small_btn(a_top, "Open web dashboard", lambda: webbrowser.open("http://localhost:5000"), width=140).pack(side="right")
level_filter = ctk.CTkOptionMenu(a_top, values=["All levels", "Critical", "High", "Medium", "Low"], width=120, height=28,
                                 fg_color=PANEL2, button_color=PANEL2, command=lambda _v: render_alerts_page())
level_filter.pack(side="right", padx=8)
a_count = ctk.CTkLabel(a_top, text="", text_color=MUTED)
a_count.pack(side="right", padx=8)
a_table = ctk.CTkScrollableFrame(alerts_page, fg_color="transparent")
a_table.pack(fill="both", expand=True, padx=10, pady=(0, 12))


def open_incident(r):
    p = evidence_file(r)
    if os.path.exists(p):
        open_path(p)
    else:
        messagebox.showwarning("Evidence", "The evidence file for this alert was not found on disk.")


def flag_false_alarm(r):
    if messagebox.askyesno("Mark as false alarm", "Mark this incident as not a threat?\nIt is kept so it can be used to retrain the scene model."):
        mark_false_alarm(r[0])
        render_alerts_page()


def render_alerts_page():
    lvl = level_filter.get()
    rows = get_incidents(level=None if lvl == "All levels" else lvl)[:PAGE_LIMIT]
    latest = get_recent_incidents(1)
    state["alerts_top"] = latest[0][0] if latest else None
    for w in a_table.winfo_children():
        w.destroy()
    a_count.configure(text=f"latest {len(rows)} (max {PAGE_LIMIT}; web dashboard has the full list)")
    if not rows:
        ctk.CTkLabel(a_table, text=EMPTY_MSG, text_color=MUTED, justify="center").pack(pady=50)
        return
    hdr = ctk.CTkFrame(a_table, fg_color="transparent")
    hdr.pack(fill="x")
    for name, w in COLS:
        ctk.CTkLabel(hdr, text=name.upper(), width=w, anchor="w", font=("Segoe UI", 10, "bold"), text_color=MUTED).pack(side="left", padx=(8, 0))
    for r in rows:
        row = ctk.CTkFrame(a_table, fg_color=PANEL2, corner_radius=8)
        row.pack(fill="x", pady=2)
        vals = [fmt_ts(r[2], "%d %b %H:%M:%S"), f"Cam {r[1]}", r[3], f"{r[4]:.0f}", r[5], f"{r[6]:.2f}", f"{r[7]:.2f}"]
        for (name, w), v in zip(COLS, vals):
            ctk.CTkLabel(row, text=v, width=w, anchor="w", font=("Segoe UI", 12, "bold" if name == "Level" else "normal"),
                         text_color=LEVEL_COLOR.get(r[3], TEXT) if name == "Level" else TEXT).pack(side="left", padx=(8, 0), pady=6)
        small_btn(row, "▶ View", lambda r=r: open_incident(r), width=70, primary=True).pack(side="right", padx=6)
        if len(r) > 10 and r[10]:
            ctk.CTkLabel(row, text="marked false alarm", text_color=MUTED, font=("Segoe UI", 11)).pack(side="right", padx=8)
        else:
            small_btn(row, "Not a threat", lambda r=r: flag_false_alarm(r), width=95).pack(side="right")


# ----- Evidence Library page -----
evidence_page, e_top = page_shell("📁  Evidence Library")
pages["evidence"] = evidence_page
small_btn(e_top, "Open folder", lambda: open_path(EVIDENCE_BLUR), width=100).pack(side="right")
e_count = ctk.CTkLabel(e_top, text="", text_color=MUTED)
e_count.pack(side="right", padx=10)
ctk.CTkLabel(evidence_page, text="Face-blurred copies only. Unblurred footage is kept separately and is available to admins through the web dashboard.",
             font=("Segoe UI", 11), text_color=MUTED, anchor="w").pack(fill="x", padx=18)
e_list = ctk.CTkScrollableFrame(evidence_page, fg_color="transparent")
e_list.pack(fill="both", expand=True, padx=10, pady=(6, 12))


def render_evidence_page():
    files = [f for f in os.listdir(EVIDENCE_BLUR) if f.endswith(".mp4")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(EVIDENCE_BLUR, f)), reverse=True)
    for w in e_list.winfo_children():
        w.destroy()
    e_count.configure(text=f"{len(files)} clip(s) · {evidence_size()} stored (blurred + raw)")
    if not files:
        ctk.CTkLabel(e_list, text="No evidence clips saved yet.", text_color=MUTED).pack(pady=50)
        return
    for f in files[:PAGE_LIMIT]:
        p = os.path.join(EVIDENCE_BLUR, f)
        row = ctk.CTkFrame(e_list, fg_color=PANEL2, corner_radius=8)
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text="🎞", font=("Segoe UI Emoji", 18)).pack(side="left", padx=(10, 6), pady=8)
        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", pady=6)
        ctk.CTkLabel(info, text=f, font=("Segoe UI", 12, "bold"), text_color=TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(info, text=f"{datetime.fromtimestamp(os.path.getmtime(p)).strftime('%d %b %Y %H:%M:%S')}  ·  {os.path.getsize(p) / 1e6:.1f} MB",
                     font=("Segoe UI", 11), text_color=MUTED, anchor="w").pack(anchor="w")
        small_btn(row, "⬇ Download", lambda p=p: save_copy(p), width=95).pack(side="right", padx=(0, 8))
        small_btn(row, "▶ Open", lambda p=p: open_path(p), width=70, primary=True).pack(side="right", padx=6)


# ----- Settings page -----
settings_page, s_top = page_shell("⚙  Settings")
pages["settings"] = settings_page
s_body = ctk.CTkScrollableFrame(settings_page, fg_color="transparent")
s_body.pack(fill="both", expand=True, padx=8, pady=(0, 10))


def section(title):
    box = ctk.CTkFrame(s_body, corner_radius=12, fg_color=PANEL2, border_width=1, border_color=BORDER)
    box.pack(fill="x", padx=8, pady=6)
    ctk.CTkLabel(box, text=title, font=("Segoe UI", 13, "bold"), text_color=TEXT).pack(anchor="w", padx=14, pady=(10, 4))
    return box


def slider_row(parent, label, lo, hi, steps, key, fmt, hint):
    r = ctk.CTkFrame(parent, fg_color="transparent")
    r.pack(fill="x", padx=14, pady=4)
    head = ctk.CTkFrame(r, fg_color="transparent")
    head.pack(fill="x")
    ctk.CTkLabel(head, text=label, font=("Segoe UI", 12), text_color=TEXT).pack(side="left")
    val = ctk.CTkLabel(head, text=fmt(SETTINGS[key]), font=("Segoe UI", 12, "bold"), text_color=ACCENT)
    val.pack(side="right")

    def on_change(v):
        SETTINGS[key] = round(v, 2)
        val.configure(text=fmt(SETTINGS[key]))
        save_settings()

    sl = ctk.CTkSlider(r, from_=lo, to=hi, number_of_steps=steps, command=on_change)
    sl.set(SETTINGS[key])
    sl.pack(fill="x", pady=(4, 0))
    ctk.CTkLabel(r, text=hint, font=("Segoe UI", 11), text_color=MUTED, anchor="w", wraplength=650, justify="left").pack(fill="x")


det = section("Detection (changes apply immediately)")
slider_row(det, "Seconds a threat must persist before recording", 1, 10, 18, "confirm", lambda v: f"{v:.1f} s",
           "Higher = fewer recordings triggered by brief flickers, but real incidents are confirmed later.")
slider_row(det, "Scene model strictness", 0.5, 0.95, 9, "scene_thresh", lambda v: f"{v:.2f}",
           "The scene model's score is heavily discounted below this value. Raise it if you get false alarms from the scene model; lower it if real fires/crashes get missed.")
multi_var = ctk.BooleanVar(value=SETTINGS["multi_person"])


def on_multi():
    SETTINGS["multi_person"] = bool(multi_var.get())
    save_settings()


ctk.CTkSwitch(det, text="Track every person in view (off = follow only the largest person)", variable=multi_var, command=on_multi).pack(anchor="w", padx=14, pady=(8, 0))
onset_var = ctk.BooleanVar(value=SETTINGS["scene_onset"])


def on_onset():
    SETTINGS["scene_onset"] = bool(onset_var.get())
    save_settings()


ctk.CTkSwitch(det, text="Scene model: alert only on sudden changes (ignores empty or look-alike scenes that never change)",
              variable=onset_var, command=on_onset).pack(anchor="w", padx=14, pady=(8, 0))
voice_var = ctk.BooleanVar(value=SETTINGS["voice"])


def on_voice():
    SETTINGS["voice"] = bool(voice_var.get())
    save_settings()


ctk.CTkSwitch(det, text="Spoken voice alerts", variable=voice_var, command=on_voice).pack(anchor="w", padx=14, pady=(6, 12))

em = section("Alerts: voice, email, Telegram and SMS")
email_ready = config.SENDER_EMAIL != "youraccount@gmail.com" and "xxxx" not in config.APP_PASSWORD
ctk.CTkLabel(em, text=f"Alerts are sent to: {config.RECEIVER_EMAIL}" + ("" if email_ready else "\nEmail is not configured yet. Set SENDER_EMAIL, APP_PASSWORD and RECEIVER_EMAIL in config.py."),
             font=("Segoe UI", 11), text_color=MUTED if email_ready else ORANGE, justify="left", anchor="w").pack(fill="x", padx=14)
sms_lbl = ctk.CTkLabel(em, text=sms_status_text(), font=("Segoe UI", 11), text_color=MUTED, anchor="w", wraplength=650, justify="left")
sms_lbl.pack(fill="x", padx=14, pady=(4, 0))
tg_lbl = ctk.CTkLabel(em, text=tg_status_text(), font=("Segoe UI", 11), text_color=MUTED, anchor="w", wraplength=650, justify="left")
tg_lbl.pack(fill="x", padx=14, pady=(4, 0))
em_status = ctk.CTkLabel(em, text="", font=("Segoe UI", 11), text_color=MUTED, anchor="w", wraplength=650, justify="left")


def test_email():
    em_status.configure(text="Sending...", text_color=MUTED)

    def work():
        try:
            send_email_alert(config.SENDER_EMAIL, config.APP_PASSWORD, config.RECEIVER_EMAIL, "SafeVision AI test alert", "This is a test alert from SafeVision AI.")
            msg, col = "✔ Test email sent", GREEN
        except Exception as e:
            msg, col = f"✖ Failed: {e}", RED
        try:
            app.after(0, lambda: em_status.configure(text=msg, text_color=col))
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


def test_sms():
    if not sms_ready():
        return em_status.configure(text=sms_status_text(), text_color=ORANGE)
    em_status.configure(text="Sending test SMS...", text_color=MUTED)

    def work():
        accepted, errors = send_sms_alert(config.TWILIO_SID, config.TWILIO_TOKEN, config.TWILIO_FROM, config.SMS_RECIPIENTS,
                                          "SafeVision AI test SMS: SMS alerts are working.")
        if accepted:
            msg, col = f"✔ Twilio accepted the test SMS for {accepted} number(s); delivery to the phone can take a minute." + (f" Problems: {errors}" if errors else ""), (ORANGE if errors else GREEN)
        else:
            msg, col = f"✖ Failed: {errors}", RED
        try:
            app.after(0, lambda: em_status.configure(text=msg, text_color=col))
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


def test_telegram():
    if not tg_ready():
        return em_status.configure(text=tg_status_text(), text_color=ORANGE)
    em_status.configure(text="Sending test message to Telegram...", text_color=MUTED)

    def work():
        delivered, errors = send_telegram_message(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_IDS, "SafeVision AI test: Telegram alerts are working.")
        if delivered:
            msg, col = f"✔ Delivered to {delivered} chat(s). Check Telegram." + (f" Problems: {errors}" if errors else ""), (ORANGE if errors else GREEN)
        else:
            msg, col = f"✖ Failed: {errors}", RED
        try:
            app.after(0, lambda: em_status.configure(text=msg, text_color=col))
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


def find_chats():
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    if not token:
        return em_status.configure(text="Put your bot token into TELEGRAM_BOT_TOKEN in config.py first.", text_color=ORANGE)
    em_status.configure(text="Looking for chats that wrote to your bot...", text_color=MUTED)

    def work():
        chats, err = telegram_find_chats(token)
        if err:
            msg, col = f"✖ Telegram said: {err}", RED
        elif not chats:
            msg, col = ("No chats found yet. In Telegram open your bot and press Start (or send it any message). For a group: add the bot, then type /start@YourBotName. "
                        "Then press this button again."), ORANGE
        else:
            msg, col = "Found: " + "; ".join(f"{cid} = {name}" for cid, name in chats.items()) + ". Put the numbers you want into TELEGRAM_CHAT_IDS in config.py.", GREEN
        try:
            app.after(0, lambda: em_status.configure(text=msg, text_color=col))
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


btn_row = ctk.CTkFrame(em, fg_color="transparent")
btn_row.pack(fill="x", padx=14, pady=8)
small_btn(btn_row, "Send test email", test_email, width=120).pack(side="left")
small_btn(btn_row, "Test voice alert", lambda: threading.Thread(target=speak_alert, args=("This is a test of the SafeVision voice alert.",), daemon=True).start(), width=120).pack(side="left", padx=8)
small_btn(btn_row, "Send test SMS", test_sms, width=120).pack(side="left", padx=8)
btn_row2 = ctk.CTkFrame(em, fg_color="transparent")
btn_row2.pack(fill="x", padx=14, pady=(0, 8))
small_btn(btn_row2, "Find chat IDs", find_chats, width=120).pack(side="left")
small_btn(btn_row2, "Send test Telegram", test_telegram, width=150).pack(side="left", padx=8)
em_status.pack(fill="x", padx=14, pady=(0, 10))

cam_box = section("Cameras")
cam_list = ctk.CTkFrame(cam_box, fg_color="transparent")
cam_list.pack(fill="x", padx=14)
cam_foot = ctk.CTkFrame(cam_box, fg_color="transparent")
cam_foot.pack(fill="x", padx=14, pady=(10, 4))
small_btn(cam_foot, "Reload config.py", lambda: reload_config(), width=130, primary=True).pack(side="left")
small_btn(cam_foot, "Scan for cameras", lambda: scan_cameras(), width=130).pack(side="left", padx=8)
small_btn(cam_foot, "Open config.py", lambda: open_in_editor(CONFIG_PATH), width=120).pack(side="left")
cam_status = ctk.CTkLabel(cam_box, text="Edits to config.py are applied automatically a moment after you save it. No restart needed.",
                          font=("Segoe UI", 11), text_color=MUTED, anchor="w", wraplength=650, justify="left")
cam_status.pack(fill="x", padx=14, pady=(0, 10))


def render_settings_cameras():
    for w in cam_list.winfo_children():
        w.destroy()
    live = [t for t in tiles if t.cam]
    if not live:
        ctk.CTkLabel(cam_list, text="No cameras found.", text_color=MUTED).pack(anchor="w")
    for t in live:
        c = t.cam
        src = re.sub(r"//[^@/]+@", "//***@", str(c["source"]))  # never show RTSP passwords on screen
        r = ctk.CTkFrame(cam_list, fg_color="transparent")
        r.pack(fill="x", pady=2)
        ctk.CTkLabel(r, text=f"Camera {c['id']}  ·  {cam_type(c['source'])}  ·  {c['location'].replace('_', ' ').title()}  ·  source: {src}",
                     font=("Segoe UI", 12), text_color=TEXT).pack(side="left")
        ctk.CTkLabel(r, text="● Online" if t.online else "● Retrying", font=("Segoe UI", 11, "bold"),
                     text_color=GREEN if t.online else ORANGE).pack(side="right")


sysbox = section("System")
for line in (f"Inference device: {device}", f"CPU threads for PyTorch: {torch.get_num_threads()}", f"Evidence folder: {os.path.join(ROOT, 'evidence')}",
             f"Settings file: {SETTINGS_PATH}"):
    ctk.CTkLabel(sysbox, text=line, font=("Segoe UI", 11), text_color=MUTED, anchor="w").pack(fill="x", padx=14)
ctk.CTkFrame(sysbox, height=8, fg_color="transparent").pack()


def show_page(name):
    state["page"] = name
    VISIBLE["dash"] = name == "dashboard"
    for k, p in pages.items():
        if k == name:
            p.grid()
        else:
            p.grid_remove()
    for k, b in nav_buttons.items():
        b.configure(fg_color=ACCENT if k == name else "transparent")
    if name == "alerts":
        render_alerts_page()
    elif name == "evidence":
        render_evidence_page()
    elif name == "settings":
        render_settings_cameras()
        sms_lbl.configure(text=sms_status_text())
        tg_lbl.configure(text=tg_status_text())


# ----- live camera management: apply config.py edits without restarting -----
def run_bg(fn, done):
    def work():
        result = fn()
        try:
            app.after(0, lambda: done(result))
        except Exception:
            pass  # window already closed

    threading.Thread(target=work, daemon=True).start()


def note(text, color=MUTED):
    cam_note.configure(text=text if len(text) < 70 else text[:67] + "...", text_color=color)
    cam_status.configure(text=text, text_color=color)
    print(f"[cameras] {text}")


def apply_cameras(new_cams):
    """Make the grid match new_cams: unchanged tiles keep running, edited ones update in place, changed or new ones restart."""
    global cameras
    cameras = new_cams[:GRID_SLOTS]
    cam_by_id.clear()
    cam_by_id.update({str(c["id"]): c for c in cameras})
    wanted = cameras + [None] * (GRID_SLOTS - len(cameras))
    for i, want in enumerate(wanted):
        old = tiles[i]
        if old.cam == want:
            continue
        if old.cam and want and old.cam["id"] == want["id"] and old.cam["source"] == want["source"]:
            old.update_cam(want)
            continue
        old.stop_async()
        old.destroy()
        tiles[i] = CameraTile(grid, want, i + 1, DW, DH)
        tiles[i].grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="nsew")
    if state["page"] == "settings":
        render_settings_cameras()


def scan_cameras():
    room = GRID_SLOTS - len(cameras)
    if room <= 0:
        return note(f"All {GRID_SLOTS} camera slots are in use.", ORANGE)
    note("Scanning for cameras...")
    used = {c["source"] for c in cameras if isinstance(c["source"], int)}

    def done(found):
        ids = {int(c["id"]) for c in cameras if str(c["id"]).isdigit()}
        new = []
        for c in found[:room]:
            ids.add(max(ids, default=0) + 1)
            new.append({**c, "id": max(ids)})
        if new:
            apply_cameras(cameras + new)
            note(f"Found {len(new)} new camera(s). They run for this session; add them to config.py to keep them.", GREEN)
        else:
            note("No new cameras found.")

    run_bg(lambda: auto_detect_cameras(skip=used), done)


def reload_config():
    try:
        ns = runpy.run_path(CONFIG_PATH)  # re-runs the file itself, so a fresh edit is always picked up
    except Exception as e:
        return note(f"config.py has an error, keeping the current cameras: {e}", RED)
    for k, v in ns.items():
        if k.isupper():
            setattr(config, k, v)  # email settings etc. update live too
    sms_lbl.configure(text=sms_status_text())
    tg_lbl.configure(text=tg_status_text())
    cams = list(ns.get("CAMERAS", []))
    stamp = datetime.now().strftime("%H:%M:%S")
    if cams:
        apply_cameras(cams)
        extra = f" Only the first {GRID_SLOTS} are shown." if len(cams) > GRID_SLOTS else ""
        note(f"config.py reloaded at {stamp}: {min(len(cams), GRID_SLOTS)} camera(s).{extra}", GREEN)
    else:
        note(f"config.py reloaded at {stamp}: no cameras listed, looking for local webcams...")
        scan_cameras()


try:
    state["cfg_seen"] = os.path.getmtime(CONFIG_PATH)
except OSError:
    pass


# ----- periodic refresh (main thread only) -----
proc = psutil.Process()
psutil.cpu_percent(None)  # prime the counters; the first call always returns 0
proc.cpu_percent(None)


def refresh():
    state["tick"] += 1
    now = datetime.now()
    live = [t for t in tiles if t.cam]
    online = sum(1 for t in live if t.online)
    crit = sum(1 for t in live if t.level == "Critical")
    high = sum(1 for t in live if t.level == "High")

    try:  # pick up edits to config.py shortly after it is saved (waits 1 s so half-written files are skipped)
        mt = os.path.getmtime(CONFIG_PATH)
        if mt != state["cfg_seen"]:
            state["cfg_seen"], state["cfg_due"] = mt, time.time() + 1.0
        elif state["cfg_due"] and time.time() >= state["cfg_due"]:
            state["cfg_due"] = None
            reload_config()
            live = [t for t in tiles if t.cam]
            online = sum(1 for t in live if t.online)
    except OSError:
        pass
    if state["page"] == "settings" and state["tick"] % 2 == 0:
        render_settings_cameras()

    cards["cams"].set(str(len(live)), f"● {online} Online", GREEN if live and online == len(live) else ORANGE)
    cards["alerts"].set(str(crit + high), f"{crit} Critical · {high} High", RED if crit else MUTED)
    cards["time"].set(now.strftime("%H:%M:%S"), "", title=now.strftime("%a, %d %b %Y"))
    for t in live:
        t.time_lbl.configure(text=now.strftime("%H:%M:%S"))

    if state["tick"] % 5 == 1:
        today = count_today_incidents()
        cards["threats"].set(str(today), "today", MUTED)
        cards["evidence"].set(evidence_size(), "blurred + raw clips", MUTED)
        bell.configure(text=f"🔔 {today}" if today else "🔔")
        render_alerts()

    if state["page"] == "alerts" and state["tick"] % 3 == 0:  # new incident arrived while looking at the Alerts page
        latest = get_recent_incidents(1)
        if (latest[0][0] if latest else None) != state["alerts_top"]:
            render_alerts_page()

    fps_vals = [t.fps for t in live if t.online and t.fps]
    all_ok = bool(live) and online == len(live)
    cpu, ram = psutil.cpu_percent(None), proc.memory_info().rss / 1e6
    app_cpu = proc.cpu_percent(None) / (os.cpu_count() or 1)  # share of the whole machine used by this app alone
    put(sys_state, "● All Systems Operational" if all_ok else "● Check Cameras", text_color=GREEN if all_ok else ORANGE)
    put(sys_vals["Cameras Online"], f"{online}/{len(live)}")
    put(sys_vals["Processing"], f"{sum(fps_vals) / len(fps_vals):.1f} FPS" if fps_vals else "-")
    put(sys_vals["PC CPU"], f"{cpu:.0f}%")
    put(sys_vals["App CPU"], f"{app_cpu:.0f}%")
    put(sys_vals["Memory Usage"], f"{psutil.virtual_memory().percent:.0f}%")
    put(sys_vals["App RAM"], f"{ram:.0f} MB")
    if state["tick"] % 30 == 0:  # lets you see in the console whether RAM keeps climbing over time
        print(f"[health] app RAM {ram:.0f} MB | threads {threading.active_count()} | PC cpu {cpu:.0f}% | app cpu {app_cpu:.0f}% | cam fps {[round(t.fps, 1) for t in live]}")
    app.after(1000, refresh)


def on_close():
    print("Shutting down... finishing any in-progress evidence saves")
    for t in tiles:
        t.running = False
    for t in tiles:
        t.shutdown()
    app.destroy()


app.protocol("WM_DELETE_WINDOW", on_close)
show_page("dashboard")
refresh()
app.mainloop()