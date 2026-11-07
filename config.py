"""
config.py
---------
All the settings in ONE place.

If the app behaves wrongly, change a number here instead of digging
through main.py. Most tuning happens in MATCH_THRESHOLD.
"""

import os

# ---------------- Camera ----------------
CAMERA_INDEX = 0            # 0 = default webcam (try 1 if you have several cameras)
FRAME_WIDTH = 1280          # higher = sharper picture, but a little slower
FRAME_HEIGHT = 720

# ---------------- Face detection (MTCNN: "where is a face?") ----------------
FACE_MIN_CONFIDENCE = 0.90  # ignore faces the detector is less than 90% sure about
MIN_FACE_SIZE = 60          # ignore faces smaller than 60 pixels (too blurry to recognize)

# ---------------- Face recognition (InceptionResnetV1: "who is it?") ----------------
RECOGNITION_MODEL = "vggface2"  # pretrained weights that come with facenet_pytorch

# How different two fingerprints can be and still count as the same person.
#   Smaller = stricter  -> fewer wrong names, but more "Unknown"
#   Bigger  = looser    -> more names found, but higher risk of mixing people up
# Start here, then tune using the distance numbers shown on screen (SHOW_DISTANCE).
MATCH_THRESHOLD = 0.9

# Voting: the name shown is the most common answer over the last N frames.
# This stops 
# Show the distance number next to each name (useful while tuning, turn off later)
SHOW_DISTANCE = True

# ---------------- Enrollment (adding people) ----------------
MIN_SNAPSHOTS = 3           # how many 'c' presses are needed before you can save a person

# ---------------- Object detection (YOLO: "what else is in the room?") ----------------
YOLO_MODEL_PATH = "yolo26n.pt"
YOLO_CONFIDENCE = 0.50
SHOW_PEOPLE_FROM_YOLO = False  # False = no big body boxes, only face boxes

# ---------------- Storage ----------------
DB_DIR = "faces"
DB_FILE = os.path.join(DB_DIR, "database.json")
