import cv2
import numpy as np
from ultralytics import YOLO

pose_model = YOLO("yolov8n-pose.pt")


def video_to_pose_sequence(video_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // seq_len, 1) if total > 0 else 1

    sequence = []
    prev_center = None
    i = 0
    while cap.isOpened() and len(sequence) < seq_len:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            kp = np.zeros(34)
            result = pose_model(frame, verbose=False)[0]
            if result.keypoints is not None and len(result.keypoints.xyn) > 0:
                people = result.keypoints.xyn.cpu().numpy()
                centers = people.mean(axis=1)
                choice = 0 if prev_center is None else np.linalg.norm(centers - prev_center, axis=1).argmin()
                kp = people[choice].flatten()
                prev_center = centers[choice]
            sequence.append(kp)
        i += 1
    cap.release()

    sequence = np.array(sequence)
    if len(sequence) < seq_len:
        pad = np.zeros((seq_len - len(sequence), 34))
        sequence = np.vstack([sequence, pad]) if len(sequence) else pad
    return sequence


def normalize_and_motion(seq):
    seq = seq.reshape(seq.shape[0], 17, 2)
    hips = (seq[:, 11] + seq[:, 12]) / 2
    centered = seq - hips[:, None, :]
    velocity = np.diff(centered, axis=0, prepend=centered[:1])
    combined = np.concatenate([centered, velocity], axis=-1)
    return combined.reshape(seq.shape[0], -1)


def extract_summary_features_v2(seq):
    seq = seq.reshape(30, 17, 4)
    velocity = seq[:, :, 2:]
    position = seq[:, :, :2]
    upper_body = velocity[:, :11, :]
    lower_body = velocity[:, 11:, :]
    return np.array([
        np.abs(velocity).mean(),
        np.abs(velocity).std(),
        np.abs(velocity).max(),
        position.std(),
        np.abs(velocity).mean(axis=(1, 2)).std(),
        np.abs(upper_body).mean(),
        np.abs(lower_body).mean(),
        np.abs(velocity).mean(axis=0).std(),
    ])


def get_features_from_video(video_path):
    seq = video_to_pose_sequence(video_path)
    normed = normalize_and_motion(seq)
    return extract_summary_features_v2(normed).reshape(1, -1)


def extract_frames(video_path, n_frames=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames
