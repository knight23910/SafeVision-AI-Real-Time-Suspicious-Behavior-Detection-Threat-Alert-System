import cv2, os, threading

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
face_net = cv2.dnn.readNetFromCaffe(os.path.join(MODELS_DIR, "deploy.prototxt"), os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"))

_net_lock = threading.Lock()  # cv2.dnn nets aren't thread-safe; two cameras can save evidence at once


def blur_faces(frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    with _net_lock:
        face_net.setInput(blob)
        detections = face_net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:  # degenerate box (e.g. at frame edge) — nothing to blur
            continue
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (35, 35), 30)
    return frame