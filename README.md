# SafeVision AI — Real-Time Suspicious Behavior Detection & Threat Alert System

A multi-camera system that watches live video, flags suspicious human behavior and scene-level anomalies (fights, robberies, explosions, fires, road accidents…), scores the risk, raises voice and email alerts, and automatically saves privacy-aware evidence clips for human review. It is built to run on ordinary hardware (a laptop CPU, no GPU required).

> **Status: working prototype.** Developed and tested on one Windows laptop (Python 3.12, Intel i5 10th gen, 8 GB RAM, no GPU) with one webcam. Phone/IP cameras, RTSP CCTV, several simultaneous cameras and low-end hardware (Raspberry Pi, old laptops) are supported by the code but **have not been tested yet**. Read [Known limitations](#known-limitations) before relying on it.

## Contents
[Features](#features) · [How a decision is made](#how-a-decision-is-made) · [Models](#models) · [Architecture](#architecture) · [Project structure](#project-structure) · [Setup](#setup) · [Configuration](#configuration) · [Usage](#usage) · [Performance & tuning](#performance--tuning) · [Troubleshooting](#troubleshooting) · [Known limitations](#known-limitations) · [Privacy & ethics](#privacy--ethics) · [Future work](#future-work) · [Tech stack & licenses](#tech-stack--licenses)

## Features

- **Two specialised models** — a pose-based *behavior* model (human actions) and an image-based *scene* model (fire, explosions, crashes…), each used where it is actually strong.
- **Operator console (desktop app)** — live camera grid with a box, ID and risk label for each person, stat cards, recent alerts, alert details with video preview, and a system-status panel (cameras online, FPS, CPU, RAM). Separate **Alerts**, **Evidence Library** and **Settings** pages.
- **Confirmation before recording** — a threat must stay High/Critical for several seconds (default 4 s, adjustable) and be re-checked repeatedly before it counts. One noisy frame never triggers a recording.
- **Evidence capture** — 5 s before + 10 s after the trigger, saved as two H.264 MP4 copies: **face-blurred** (default access) and **raw** (admin only), plus JSON metadata and a database record.
- **Alerts** — offline spoken voice alert when an incident is confirmed, an email with the incident details to one or several addresses (Gmail SMTP), and optionally **Telegram** messages (free, can include the face-blurred clip) and SMS (Twilio, paid). Email, Telegram and SMS need internet.
- **Web dashboard** — login, roles (admin sees raw footage, guard sees blurred only), incident stats, level filter, in-browser playback.
- **False-alarm feedback** — mark an incident *Not a threat* (desktop Alerts page or web dashboard) and export those frames to retrain the scene model later. Learning is human-approved and offline by design, never silent self-modification.
- **Multi-camera, config-driven** — USB/laptop webcams, phone cameras (IP-camera apps) and RTSP CCTV through one `config.py` list; local webcams are auto-detected if the list is empty. Dropped streams reconnect automatically.
- **Runs on CPU** — frame skipping, small model inputs, JPEG-compressed evidence buffers and light video encoding keep RAM and CPU low.

## How a decision is made

1. Each camera runs in its own thread. A YOLOv8-Pose model finds every person in view (up to 6 per camera) and gives their keypoints. Each person is followed with their own ID, and their own last 30 pose samples (≈5 s) are summarised into 8 motion features for the behavior model. The most suspicious person sets the camera's behavior score. The scene model scores a downscaled frame every ~0.5 s.
2. The **higher** of the two confidences becomes the base score: below 0.5 maps to 0–35, above 0.5 maps to 35–100. The scene model's score is heavily discounted (×0.3) unless it exceeds the *scene strictness* threshold (default 0.85), which suppresses most false alarms from unusual lighting or framing. On top of that, a high scene score only counts if it **just jumped above that camera's own recent normal** (median of the last 3 minutes, first 20 s are calibration). An empty room that always scores high never jumps and is ignored; a fire or explosion appears suddenly and passes. This can be switched off in Settings.
3. The score is multiplied by a **location factor** (×1.3 for `atm`, `gate`, `server_room`) and a **time factor** (×1.2 at night, 20:00–06:00 when `time_of_day` is `"auto"`), capped at 100.

| Score | Level | Meaning |
|---|---|---|
| < 35 | Low | normal |
| 35 – 59 | Medium | watch |
| 60 – 79 | High | alert-worthy |
| ≥ 80 | Critical | alert-worthy |

4. If High/Critical persists for the confirmation time (default 4 s) and has been re-checked at least 4 times, the incident is **confirmed**: voice alert fires, evidence is recorded, the incident is logged, and the email is sent. A new incident can only start after the risk has dropped back below High.

## Models

| | Behavior model | Scene model |
|---|---|---|
| Detects | fighting, assault, abuse, robbery, shooting, arrest, shoplifting (reported as one "suspicious" class) | explosion, arson, burglary, road accidents, vandalism, stealing (reported as one "anomaly" class) |
| Input | 30-sample pose window → 8 hand-crafted motion features | single frame (224×224) |
| Method | Random Forest on YOLOv8n-Pose keypoints | MobileNetV2, ImageNet transfer learning |
| Measured accuracy | **≈71–74%** (5-fold cross-validation, depending on the data subset) | **≈98%** on a held-out, video-level split (includes 100% on the phone-style normal clips in that split) |

Both models are binary (normal vs. suspicious); they do **not** name the specific behavior. The UI therefore says "Suspicious" / "Anomaly", not "Fighting".

**Data and training.** Trained in Kaggle notebooks (not included in this repo) on the *UCF-Crime* dataset (Kaggle mirror "Anomaly Detection Dataset UCF") for the suspicious classes and normal footage, plus the Kaggle *Human Activity & Suspicious Behavior Video Dataset* for normal everyday activities (walking, sitting, standing, clapping, …). The scene model was retrained with that second dataset after early webcam tests showed it labelling normal close-range webcam frames as threats; train/validation splits are done **per video** to avoid frame leakage. The accuracy figures above come from this project's own validation and are not a guarantee of real-world performance.

## Architecture

```
Camera (webcam / phone / CCTV)
        │
        ▼
 capture thread (one per camera)
   ├─► YOLOv8-Pose (track + keypoints) ─► 8 motion features ─► Behavior model ─┐
   └─► MobileNetV2 scene classifier ────────────────────────────────────────────┤
                                                                                 ▼
                                              risk score × location × time  →  Low / Medium / High / Critical
                                                                                 │ must stay High/Critical for N seconds
                                       ┌─────────────────────────────┬───────────┴──────────┐
                                       ▼                             ▼                      ▼
                                voice alert (offline)         email alert        evidence: JPEG buffer → raw + blurred MP4
                                                                                        + JSON + SQLite record
                                                                                             │
                                          operator console (desktop)  ◄─────────────────────┴──────► web dashboard (login, roles)
```

## Project structure

```
suspicious-behavior-detector/
├── app/
│   ├── gui.py        # main operator console — run this
├── dashboard/
│   └── dashboard_app.py         # Flask web dashboard (login, roles, incidents, playback)
├── modules/
│   ├── __init__.py
│   ├── pose_utils.py            # pose extraction + feature engineering
│   ├── scene_utils.py           # scene model loading + inference
│   ├── face_blur.py             # DNN face detection + blurring
│   ├── alert_utils.py           # email + voice alerts
│   ├── db_utils.py              # SQLite incident log
│   └── pipeline.py              # risk scoring + evidence saving
├── models/                      # model files (see Setup — not in the repo)
├── config.py                    # cameras, email credentials, user accounts  (edit this; keep it private)
├── requirements.txt
├── settings.json                # created at runtime: values changed on the Settings page
├── incidents.db                 # created at runtime: incident log
├── evidence/                    # created at runtime: raw/ and blurred/ clips
└── feedback_normal_frames/      # created by tests/export_false_alarms.py
```

## Setup

Developed on **Windows with Python 3.12**. Other platforms are untested.

1. **Create an environment and install dependencies** (from the project root):
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```
   OpenCV is pinned on purpose: `opencv-python` 5.x caused errors during testing. Run `pip freeze` once your setup works if you want an exact record of versions.

2. **Add the model files to `models/`** (not included — train them or get them from the project owner):
   - `behavior_model.pkl` — Random Forest behavior classifier
   - `scene_model.pth` — MobileNetV2 scene classifier weights
   - `yolov8n-pose.pt` — downloads automatically on first run if missing

3. **Add the face-detector files to `models/`:**
   - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

4. **Edit `config.py`** (see below), then start the operator console.

Always launch scripts **from the project root** (`python app/live_dashboard.py`).

## Configuration

`config.py`:

| Setting | What it does |
|---|---|
| `SENDER_EMAIL`, `APP_PASSWORD`, `RECEIVER_EMAIL` | Email alerts. `APP_PASSWORD` must be a Gmail **app password** (Google Account → Security → 2-Step Verification → App passwords), not your normal password. `RECEIVER_EMAIL` can hold several addresses separated by commas. |
| `USERS` | Web-dashboard logins and roles (`admin` / `guard`). **Change the defaults.** |
| `CAMERAS` | Cameras to monitor. Leave it empty to auto-detect local webcams. |
| `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS`, `TELEGRAM_MIN_LEVEL`, `TELEGRAM_COOLDOWN`, `TELEGRAM_SEND_CLIP` | Optional free Telegram alerts, see below. |
| `SMS_ENABLED`, `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM`, `SMS_RECIPIENTS`, `SMS_MIN_LEVEL`, `SMS_COOLDOWN` | Optional SMS alerts through Twilio, see below. |

```python
CAMERAS = [
    {"id": 1, "source": 0, "location": "general", "time_of_day": "auto"},                      # laptop / USB webcam
    {"id": 2, "source": "rtsp://user:pass@192.168.1.10:554/stream", "location": "atm", "time_of_day": "auto"},   # CCTV
    {"id": 3, "source": "http://192.168.1.5:8080/video", "location": "gate", "time_of_day": "auto"},            # phone (IP-camera app)
]
```
`location` of `atm`, `gate` or `server_room` raises risk ×1.3. `time_of_day` is `"day"`, `"night"` or `"auto"` (system clock).

**Camera changes apply live, with no restart.** Save `config.py` and the console picks it up about a second later (or press *Reload config.py* on the Settings page). Unchanged cameras keep running, location / time-of-day edits apply instantly, and a camera whose source changed reconnects. An unreachable camera keeps retrying every 5 s, so a USB camera you plug in later connects by itself. If `config.py` has a typo, the current cameras keep running and the error is shown on the Settings page. *Scan for cameras* adds local webcams for the current session only; add them to `config.py` to keep them. Email settings reload too.

`settings.json` is written by the **Settings** page: seconds a threat must persist before recording, scene-model strictness, scene-change-only mode, and voice alerts on/off. These take effect immediately.

Before sharing the project, do not publish `config.py` with real credentials; also change the Flask `secret_key` in `dashboard/dashboard_app.py`.

### Telegram alerts (free, no registration)

The simplest way to alert phones: no payment, no phone-network rules, and it can attach the evidence clip. A Telegram *bot* sends the messages.

1. In Telegram open **@BotFather**, send `/newbot`, choose a name and a username ending in `bot`, and copy the **token** it gives you.
2. In `config.py` set `TELEGRAM_ENABLED = True` and `TELEGRAM_BOT_TOKEN = "..."`.
3. Everyone who should receive alerts opens your bot and presses **Start** (a bot cannot message someone first). For a team group: create the group, add the bot, and type `/start@YourBotName` in it.
4. In the dashboard open **Settings** and press **Find chat IDs**. It lists every chat that wrote to the bot. Copy the numbers you want into `TELEGRAM_CHAT_IDS` (group ids are negative), e.g. `[123456789, -100123456789]`.
5. Press **Send test Telegram**.

A message is sent the moment an incident is confirmed, for `TELEGRAM_MIN_LEVEL` and above (default `"High"`), at most once per camera every `TELEGRAM_COOLDOWN` seconds. With `TELEGRAM_SEND_CLIP = True` the **face-blurred** clip (up to 50 MB, Telegram's bot limit) follows a few seconds after it has been saved; the raw clip is never sent. Bot messages are not end-to-end encrypted, and a clip sent this way is stored on Telegram's servers, which is why this is off by default. The bot token sits in plain text in `config.py`: do not publish that file.

### SMS alerts (optional, paid)

SMS goes to the phone numbers in `SMS_RECIPIENTS` the moment an incident is confirmed (together with the voice alert, not after the clip is saved). To keep cost down it is only sent for `SMS_MIN_LEVEL` and above (default `"Critical"`) and at most once per camera every `SMS_COOLDOWN` seconds (default 120). Messages are plain ASCII, about 110 characters, so each one is a single SMS.

1. Create a [Twilio](https://www.twilio.com) account and note the **Account SID** and **Auth Token**; get a Twilio phone number (or a Messaging Service, whose SID starts with `MG`).
2. In `config.py` set `SMS_ENABLED = True`, `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM`, and `SMS_RECIPIENTS = ["+919876543210", ...]` (full international format).
3. Open **Settings** and press **Send test SMS**. Settings reload automatically, no restart needed.

Things to know before relying on it:
- It needs a provider account and **costs money per message**; free trial accounts can typically only text numbers you have verified.
- *Accepted* by Twilio is not the same as *delivered*: always check that the test text actually reaches the phone.
- Business SMS to Indian numbers is regulated in India (DLT registration of sender IDs and templates). Delivery rules depend on your account and route, so test early; the Telegram option above is free and much easier to set up.
- `TWILIO_TOKEN` sits in plain text in `config.py`: do not publish that file.

## Usage

**Operator console** (the main app):
```bash
python app/live_dashboard.py
```
| Page | What you get |
|---|---|
| Dashboard | Stat cards, live camera grid (up to 4 tiles; empty slots show "Not Available"), recent alerts, alert details with preview + *View Full Video* / *Download*, system status |
| Alerts | Latest 40 incidents with a level filter, *View*, and *Not a threat*; link to the web dashboard |
| Evidence Library | Face-blurred clips with date and size, *Open* / *Download* / open folder |
| Settings | Detection sliders, switches (track every person, scene-change mode, voice), **Send test email**, **Send test Telegram**, **Find chat IDs**, **Send test SMS**, **Test voice alert**, live camera status, **Reload config.py**, **Scan for cameras**, `config.py` shortcut |

The desktop app only shows the **blurred** clips. Closing the window while a clip is being saved finishes the save first — wait for the `Evidence saved` line in the terminal.

**Web dashboard** (logins, roles, remote viewing on your network):
```bash
python dashboard/dashboard_app.py        # then open http://localhost:5000
```
Default logins in `config.py` are `admin / admin123` and `guard / guard123` — change them. It listens on all network interfaces (`0.0.0.0:5000`) without HTTPS, so only run it on a trusted network.

**Other tools**
```bash
python app/multi_gui.py                         # test detection on up to 4 video files
python app/live_monitor.py                      # older OpenCV-window monitor (press q to quit)
python tests/view_incidents.py                  # print the incident database
python tests/export_false_alarms.py             # export frames of incidents marked "Not a threat"
python capture_normal_frames.py                 # capture webcam frames of normal activity
```

**Improving the scene model over time:** mark false alarms as *Not a threat* → run `tests/export_false_alarms.py` → add the exported frames to the "Normal" class of the scene-model training set and retrain → replace `models/scene_model.pth`. No code changes are needed.

## Performance & tuning

**Measured** on the development laptop (i5 10th gen, 8 GB RAM, no GPU, one webcam, console idle): about **600 MB RAM**, about **11% of total CPU**, and 29–31 FPS from the camera. While an evidence clip is being written, usage briefly rises (about 30% CPU, +80 MB RAM) and returns to normal afterwards. These are single short runs, not benchmarks. The console prints a `[health]` line every 30 s (RAM, threads, PC CPU, app CPU, camera FPS), and the System Status panel shows the same numbers live.

**Expected, not measured:** a dual-core CPU with 4 GB RAM should be a reasonable floor; a Raspberry Pi 4 (4 GB) might manage one camera with the low-end settings below. Please test on your own hardware.

Tuning knobs (constants at the top of `app/live_dashboard.py`):

| Constant | Default | Effect |
|---|---|---|
| `FRAME_SKIP` | 5 | Pose/behavior check every Nth frame. The behavior model was trained on ~5-second windows (30 samples), so change this carefully. |
| `SCENE_SKIP` | 15 | Scene-model check every Nth frame. Safest value to raise on slow hardware. |
| `EVIDENCE_EVERY` | 2 | Keep every Nth frame in evidence clips (2 = half the camera FPS). Higher = less RAM and CPU. |
| `EVIDENCE_WIDTH` | 640 | Maximum width of stored evidence frames. |
| `GRID_SLOTS` | 4 | Number of camera tiles. |
| `MAX_PEOPLE` | 6 | Most people tracked per camera (the biggest, i.e. closest, are kept). |
| `PRE_SECONDS` / `POST_SECONDS` | 5 / 10 | Evidence before / after the trigger. |

For weak hardware: use one or two cameras, set `GRID_SLOTS` to the number you use, raise `SCENE_SKIP` first, then `EVIDENCE_EVERY` to 3 and `EVIDENCE_WIDTH` to 480.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'modules'` | Run from the project root (`python app/...`) and make sure `modules/__init__.py` exists. |
| `cv2 has no attribute 'CascadeClassifier'` or other odd OpenCV errors | You have OpenCV 5.x. `pip install opencv-python==4.10.0.84`. |
| Email fails with `535 Username and Password not accepted` | Use a Gmail **app password** (needs 2-Step Verification). Test from Settings → *Send test email*. |
| **Send test Telegram** fails | `Unauthorized` = wrong bot token. `chat not found` or `can't initiate conversation` = that person has not pressed **Start** on the bot, or the id is wrong. **Find chat IDs** shows nothing = send the bot a message first, then press it again. |
| **Send test SMS** fails | Read the error text shown under the buttons: wrong SID or token, an unverified recipient on a trial account, or a number that is not in international format (`+91...`). |
| Evidence clip will not play in the browser | Clips must be H.264. They are converted with the bundled ffmpeg (`imageio-ffmpeg`); clips saved by very old versions may not play. |
| Error mentioning the `incidents` table or `false_alarm` column | Delete `incidents.db` (old incident records are lost) and restart. |
| An empty room keeps showing Critical | Make sure Settings → *Scene model: alert only on sudden changes* is on. For a lasting fix, record empty-scene frames with `capture_normal_frames.py` (day and night), add them to the "Normal" class and retrain the scene model. |
| High CPU or growing RAM | Check the `[health]` lines and the System Status panel; apply the tuning knobs above; close other heavy apps. |
| Camera not found | Try another index (`1`, `2`) or set the stream URL, then save `config.py` (no restart needed) or use Settings → *Scan for cameras*. Close apps that hold the camera (Teams, Zoom). |
| scikit-learn warns when loading `behavior_model.pkl` | Install the scikit-learn version the model was trained with. |

## Known limitations

- **Behavior accuracy is only ≈71–74%.** It will miss some incidents and raise some false alarms. Several algorithms (Random Forest, XGBoost, SVM, logistic regression, LSTM) and a 4× larger dataset all landed in the same range, which points to the limits of the hand-crafted pose features rather than the algorithm. Richer features or better tracking are the likely path forward.
- **The scene model can still misread unusual frames** (very close, dim or low-quality). Retraining with phone-style normal footage fixed the worst case we found; the strictness threshold and the confirmation delay reduce what remains but cannot remove it.
- **Scene alerts need a sudden change.** To stop empty or constantly look-alike scenes (for example an empty office at night) from showing Critical forever, scene alerts fire only when the score jumps above the camera's own recent baseline. Consequences: a scene event in the first 20 s after start-up is not caught, a camera that starts out already looking at a fire will not alert, and a change that develops very slowly (over minutes) may be treated as normal. The proper fix is to retrain the scene model with empty-scene frames as "Normal".
- **Binary decisions only.** The system says "suspicious" or "anomaly", not which behavior occurred.
- **Tested narrowly.** One Windows laptop and one webcam. Phone cameras, RTSP CCTV, multiple simultaneous cameras and low-end hardware are untested.
- **Simple multi-person tracking.** Up to 6 people per camera are followed by nearest-position matching, so IDs can swap when people cross or hide behind each other, and a newly arrived person is not scored until their ~5 s movement history has filled. More people in view also means more chances for a false alert, because the most suspicious person sets the score. The behavior model was trained on single-person clips and has not been measured on crowds. Settings has a switch to follow only the largest person.
- **Security is demo-grade.** Passwords are plaintext in `config.py`, the web dashboard has no HTTPS, and the default logins and Flask secret key must be changed. Evidence files are not encrypted.
- **No retention policy.** Evidence and the database grow until you delete them.
- **Email, Telegram and SMS need internet**; voice alerts and detection work offline. Telegram is free. SMS needs a paid provider account (Twilio), and delivery to Indian numbers is subject to India's DLT rules.
- **Stored clips are lower quality than the camera:** at most 640 px wide, half the camera frame rate by default, JPEG-compressed in memory before being encoded.
- The face blur uses a lightweight detector and can miss small, turned or partly hidden faces — treat it as privacy-by-default, not a guarantee.

## Privacy & ethics

The system flags *patterns of movement and scenes*, not identity, intent or guilt; a human should review every alert before any action is taken. Evidence is kept in two tiers: face-blurred copies for everyday review, and raw copies restricted to the admin role. Sending the blurred clip to Telegram is off by default; if you switch it on, the clip is stored on Telegram's servers, and the raw copy is never sent anywhere. Only deploy it where you have the right to record, tell people they are being recorded, and have a policy for how long footage is kept. The training data also limits fairness: test it across clothing, body types and lighting in your own environment before trusting it.

## Future work

- Export both models to **ONNX** and run with ONNX Runtime (lower RAM and CPU), then test on a Raspberry Pi / old laptop and record real numbers.
- Name the specific behavior (multi-class) and improve tracking for crowded scenes (e.g. ByteTrack).
- Evidence retention / auto-delete, hashed passwords, HTTPS, encrypted evidence.
- Charts on the dashboard (threats over time, behavior types); WhatsApp alerts.
- Retrain the scene model with empty-scene frames (offices, corridors and car parks at night) so it no longer needs the sudden-change gate.
- Periodic retraining with the exported false-alarm feedback.

## Tech stack & licenses

Python, PyTorch, Ultralytics YOLOv8-Pose, torchvision (MobileNetV2), OpenCV, scikit-learn, Flask, SQLite, CustomTkinter, pyttsx3, psutil, imageio-ffmpeg.

- Ultralytics YOLOv8 is licensed under **AGPL-3.0**; read its terms before redistributing or deploying commercially.
- The datasets (UCF-Crime and the Kaggle Human Activity & Suspicious Behavior Video Dataset) are **not** included and have their own terms.
- No license has been chosen for this project's own code yet — add a `LICENSE` file before publishing.
