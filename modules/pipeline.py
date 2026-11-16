import cv2, os, json, subprocess
import imageio_ffmpeg
from modules.face_blur import blur_faces
from modules.db_utils import log_incident
from modules.alert_utils import send_email_alert, send_telegram_video
import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root, so evidence lands in the same place no matter where you launch from
RAW_DIR, BLUR_DIR = os.path.join(ROOT, "evidence", "raw"), os.path.join(ROOT, "evidence", "blurred")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(BLUR_DIR, exist_ok=True)


def risk_from(p):
    return (p / 0.5) * 35 if p < 0.5 else 35 + ((p - 0.5) / 0.5) * 65


def _decode(f):
    return f if f.ndim == 3 else cv2.imdecode(f, cv2.IMREAD_COLOR)  # evidence arrives as raw 3-D frames or 1-D JPEG buffers


def to_h264(temp_path, final_path):
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run([ffmpeg, "-y", "-i", temp_path, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # H.264 needs even dimensions
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-threads", "2",  # light on CPU so the live view stays smooth
                            "-pix_fmt", "yuv420p", final_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = r.returncode == 0 and os.path.exists(final_path)
    except Exception:
        ok = False
    if ok:
        os.remove(temp_path)
    else:
        os.replace(temp_path, final_path)  # conversion failed: keep the clip anyway instead of losing the evidence


def save_evidence(frames, meta, fps):
    name = f"cam{meta['camera_id']}_{meta['timestamp']}"
    h, w = _decode(frames[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_temp, blur_temp = os.path.join(RAW_DIR, f"{name}_temp.mp4"), os.path.join(BLUR_DIR, f"{name}_temp.mp4")
    raw_out = cv2.VideoWriter(raw_temp, fourcc, fps, (w, h))
    blur_out = cv2.VideoWriter(blur_temp, fourcc, fps, (w, h))
    for f in frames:  # decode one frame at a time so a long clip never has to sit in RAM uncompressed
        img = _decode(f)
        raw_out.write(img)
        blur_out.write(blur_faces(img))  # raw is already written, so blurring in place is safe
    raw_out.release()
    blur_out.release()

    to_h264(raw_temp, os.path.join(RAW_DIR, f"{name}.mp4"))
    to_h264(blur_temp, os.path.join(BLUR_DIR, f"{name}.mp4"))

    with open(os.path.join(RAW_DIR, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log_incident(meta, f"evidence/raw/{name}.mp4", f"evidence/blurred/{name}.mp4")
    print(f"🚨 Evidence saved: {name}.mp4 [{meta['level']}] (raw + blurred)")

    try:
        subject = f"[{meta['level']}] Threat detected - Camera {meta['camera_id']}"
        body = (f"Risk level: {meta['level']} ({meta['risk_score']})\n"
                f"Category: {meta['category']}\n"
                f"Behavior confidence: {meta['behavior_conf']}  Scene confidence: {meta['scene_conf']}\n"
                f"Timestamp: {meta['timestamp']}\n"
                f"Evidence file: {name}.mp4")
        send_email_alert(config.SENDER_EMAIL, config.APP_PASSWORD, config.RECEIVER_EMAIL, subject, body)
        print("📧 Alert email sent")
    except Exception as e:
        print(f"⚠️ Email alert failed: {e}")  # evidence is still saved even if email fails

    try:  # optional: the face-blurred clip goes to Telegram; the raw clip is never sent anywhere
        ranks = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        if (getattr(config, "TELEGRAM_ENABLED", False) and getattr(config, "TELEGRAM_SEND_CLIP", False)
                and getattr(config, "TELEGRAM_BOT_TOKEN", "") and getattr(config, "TELEGRAM_CHAT_IDS", [])
                and ranks.get(meta["level"], 0) >= ranks.get(getattr(config, "TELEGRAM_MIN_LEVEL", "High"), 2)):
            caption = f"Evidence: Camera {meta['camera_id']}, {meta['level']} ({meta['risk_score']}), {meta['timestamp']}. Faces blurred."
            delivered, errors = send_telegram_video(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_IDS, os.path.join(BLUR_DIR, f"{name}.mp4"), caption)
            print(f"[telegram] clip delivered to {delivered} chat(s)" + (f" | errors: {errors}" if errors else ""))
    except Exception as e:
        print(f"⚠️ Telegram clip failed: {e}")