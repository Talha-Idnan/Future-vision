"""
main.py
-------
The main program. For every webcam frame it:

  1. Finds OTHER OBJECTS with YOLO (drawn in blue, people skipped by default)
  2. Finds FACES with MTCNN (where is each face?)
  3. Turns each face into a FINGERPRINT with InceptionResnetV1 (who is it?)
  4. Compares the fingerprint with saved people (face_db.py)
  5. Draws a box and a name on each face, then shows the frame

Keys while the window is active:
  c = take a snapshot of the single face on screen (do this several times)
  n = save a new person (type a name in the terminal)
  d = delete a person (type a name in the terminal)
  q = quit
"""

import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

from face_db import FaceDatabase
from types import SimpleNamespace


config = SimpleNamespace(
    # Camera
    CAMERA_INDEX=0,
    FRAME_WIDTH=1280,
    FRAME_HEIGHT=720,

    # Face detection
    FACE_MIN_CONFIDENCE=0.90,
    MIN_FACE_SIZE=60,

    # Face recognition
    RECOGNITION_MODEL="vggface2",
    MATCH_THRESHOLD=0.9,
    VOTE_WINDOW=7,
    SHOW_DISTANCE=True,
    MIN_SNAPSHOTS=3,

    # Object detection
    YOLO_MODEL_PATH="yolo26n.pt",
    YOLO_CONFIDENCE=0.50,
    SHOW_PEOPLE_FROM_YOLO=False,

    # Storage
    DB_DIR="faces",
    DB_FILE="faces/database.json",
)


# ==============================================================
# 1. Load devices and models (once, before the camera starts)
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model A: finds WHERE faces are. Returns boxes and confidence scores.
face_detector = MTCNN(keep_all=True, device=device)

# Model B: turns a face image into a fingerprint of 512 numbers. Tells us WHO.
face_encoder = InceptionResnetV1(pretrained=config.RECOGNITION_MODEL).eval().to(device)

# Model C: YOLO, finds general objects (cup, phone, chair, and so on).
object_detector = YOLO(config.YOLO_MODEL_PATH)

# The saved album of known people.
db = FaceDatabase()

# Colors are BGR (OpenCV's order, which is blue-green-red, not red-green-blue).
COLOR_KNOWN = (120, 200, 0)     # green-ish: recognized person
COLOR_UNKNOWN = (60, 60, 230)   # red-ish: face we don't know
COLOR_OBJECT = (255, 160, 0)    # blue-ish: YOLO object


# ==============================================================
# 2. Helper functions
# ==============================================================

def make_fingerprint(face_bgr):
    """Convert one face image into a normalized list of 512 numbers."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)       # OpenCV BGR -> RGB for the model
    rgb = cv2.resize(rgb, (160, 160))                      # the model expects 160x160 pixels

    # Image -> tensor (a PyTorch array). HWC (height, width, color) -> CHW (color, height, width).
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0                      # scale pixel values to about -1 to 1

    with torch.no_grad():                                  # inference only, no learning
        embedding = face_encoder(tensor.unsqueeze(0).to(device))[0]

    embedding = embedding.cpu().numpy()
    return embedding / np.linalg.norm(embedding)           # scale so the length is exactly 1


def crop_face(frame, box, margin=0.15):
    """Cut the face out of the frame, with a small margin so hair and chin are included."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pad_x = int((x2 - x1) * margin)
    pad_y = int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)        # never go outside the image
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def draw_box(img, box, text, color):
    """Draw a thin rectangle and a small filled label bar above it (clean, modern look)."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(0, y1 - text_h - 12)
    cv2.rectangle(img, (x1, top), (x1 + text_w + 10, y1), color, -1)   # filled label bar
    cv2.putText(img, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)


class LabelSmoother:
    """
    Stops names from flickering.

    It remembers the last few labels for each face and shows the most common one.
    Faces are numbered left to right, so slot 0 is the leftmost face. This is a
    simple version of tracking. It works well for a few people sitting still.
    """

    def __init__(self, window):
        self.window = window        # how many past frames to remember
        self.slots = []             # one history (deque) per face slot

    def update(self, labels):
        """Take the labels for this frame and return the smoothed labels."""
        del self.slots[len(labels):]            # forget slots for faces that left
        smoothed = []
        for i, label in enumerate(labels):
            while len(self.slots) <= i:         # create a new slot if a face appeared
                self.slots.append(deque(maxlen=self.window))
            self.slots[i].append(label)
            most_common = Counter(self.slots[i]).most_common(1)[0][0]
            smoothed.append(most_common)
        return smoothed


def find_faces(frame):
    """
    Detect and identify all faces in one frame.
    Returns a list of dictionaries, sorted left to right.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)           # MTCNN expects RGB
    boxes, probs = face_detector.detect(rgb)

    faces = []
    if boxes is None:                                       # no faces in this frame
        return faces

    for box, prob in zip(boxes, probs):
        if prob is None or prob < config.FACE_MIN_CONFIDENCE:
            continue                                        # detector not confident enough

        x1, y1, x2, y2 = box
        if min(x2 - x1, y2 - y1) < config.MIN_FACE_SIZE:
            continue                                        # too small to recognize reliably

        crop = crop_face(frame, box)
        if crop.size == 0:
            continue

        fingerprint = make_fingerprint(crop)
        name, dist = db.identify(fingerprint)
        faces.append({"box": box, "fingerprint": fingerprint, "name": name, "dist": dist})

    faces.sort(key=lambda f: f["box"][0])                   # left to right by x position
    return faces


def draw_objects(frame):
    """Run YOLO and draw its boxes. Person boxes are skipped unless turned on in config."""
    results = object_detector(frame, conf=config.YOLO_CONFIDENCE, verbose=False)[0]
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        class_id = int(cls)
        if class_id == 0 and not config.SHOW_PEOPLE_FROM_YOLO:
            continue                                        # class 0 is "person" in COCO
        draw_box(frame, box, object_detector.names[class_id], COLOR_OBJECT)


# ==============================================================
# 3. Main loop
# ==============================================================

def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Ask the camera for a higher resolution (fixes the blurry, stretched look)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    smoother = LabelSmoother(config.VOTE_WINDOW)
    pending = []            # fingerprints captured with 'c' for the person being enrolled
    fps = 0.0               # smoothed frames per second
    window_name = "Future Vision - Face Recognition"

    print("Keys: c = capture snapshot | n = save person | d = delete person | q = quit")

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read camera frame")
            break

        frame_start = time.perf_counter()

        # Step A: objects (drawn first, so face boxes sit on top)
        draw_objects(frame)

        # Step B: faces, then smooth the names over recent frames
        faces = find_faces(frame)
        labels = smoother.update([face["name"] for face in faces])

        # Step C: draw each face with its (smoothed) name
        for face, label in zip(faces, labels):
            if label == "Unknown":
                text, color = "Unknown", COLOR_UNKNOWN
            else:
                text, color = label, COLOR_KNOWN

            if config.SHOW_DISTANCE and face["dist"] is not None:
                text += f"  {face['dist']:.2f}"          # distance helps when tuning

            draw_box(frame, face["box"], text, color)

        # Step D: measure speed, then draw the status text
        frame_time = time.perf_counter() - frame_start
        fps_now = 1.0 / max(frame_time, 1e-6)
        fps = fps_now if fps == 0 else 0.9 * fps + 0.1 * fps_now   # smooth the number

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_KNOWN, 2)
        cv2.putText(frame, f"Snapshots: {len(pending)}  |  c=capture  n=save  d=delete  q=quit",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        # Step E: keyboard
        # waitKey(1) waits 1 millisecond for a key. & 0xFF keeps only the key code.
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            # Only take a snapshot when exactly one face is visible, so we don't mix people up
            if len(faces) != 1:
                print("Snapshot skipped: show exactly one face")
            else:
                pending.append(faces[0]["fingerprint"])
                print(f"Snapshot {len(pending)} saved")

        elif key == ord("n"):
            if len(pending) < config.MIN_SNAPSHOTS:
                print(f"Need at least {config.MIN_SNAPSHOTS} snapshots (you have {len(pending)})")
            else:
                # input() pauses the window until you type in the terminal. That is normal.
                name = input("Name for this person: ").strip()
                if name:
                    db.add_person(name, pending)
                    print(f"Saved {name} with {len(pending)} snapshots")
                    pending.clear()

        elif key == ord("d"):
            print("Known people:", db.names() or "none")
            name = input("Name to delete (blank = cancel): ").strip()
            if name and db.delete_person(name):
                print(f"Deleted {name}")
            elif name:
                print(f"No one named '{name}' in the database")

    # Step F: clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
