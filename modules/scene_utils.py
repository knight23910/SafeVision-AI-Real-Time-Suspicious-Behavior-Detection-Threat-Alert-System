import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from modules.pose_utils import extract_frames

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_scene_model(weights_path, device):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(1280, 2)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m = m.to(device)
    m.eval()
    return m


def predict_scene_frame(scene_model, device, frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(scene_model(tensor), dim=1)[0, 1].item()


def predict_scene(scene_model, device, video_path, n_frames=5):
    frames = extract_frames(video_path, n_frames=n_frames)
    if not frames:
        return 0.0
    batch = []
    for f in frames:
        img = f.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
    batch = torch.stack(batch).to(device)

    with torch.no_grad():
        out = scene_model(batch)
        probs = torch.softmax(out, dim=1)[:, 1]
    return probs.max().item()